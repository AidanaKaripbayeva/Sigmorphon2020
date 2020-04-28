from data import feature_converters, SigmorphonData_Factory
from data.sigmorphon2020_data_reader import *
import logging
from models import ModelFactory
from optimizers import OptimizerFactory
import os
import pickle
import shutil
import time
import torch
import wandb

# Mark the start of the execution. This is used to generate a meaningful name for the execution.
execution_identifier = str(time.time())
if "WANDB_AUTO_IDENTIFIER" in os.environ:
    execution_identifier = str(time.time())
elif "WANDB_IDENTIFIER" in os.environ:
    execution_identifier = os.environ["WANDB_IDENTIFIER"]
elif "PBS_JOBID" in os.environ:
    execution_identifier = os.environ["PBS_JOBID"]


class Experiment:
    """
    The class steering the flow of an experiment, persisting its intermediary results and reporting its final result.
    """

    class AutoPaddedLoss(object):
        def __init__(self, loss, pad_index=Alphabet.stop_integer, device=None):
            self.loss = loss
            self.pad_index = pad_index
            self.device = device
        
        def __call__(self, probs, target):
            L_probs = probs.shape[0]
            L_target = target.shape[0]
            
            if L_probs < L_target:
                #TODO: Fewer assumptions about the shape of probs
                extra_probs = torch.zeros(L_target-L_probs, probs.shape[1],device=self.device)
                extra_probs[:,self.pad_index] = 1
                probs = torch.cat([probs, extra_probs])
                
            
            if L_target < L_probs:
                target = torch.cat([target] + [torch.full((L_probs-L_target,),self.pad_index,dtype=torch.long)])
            
            probs = probs.to(self.device)
            target = target.to(self.device)
            return self.loss(probs,target)
        
        def to(self, target_device):
            self.device = target_device
            self.loss = self.loss.to(self.device)
            return self

    def __init__(self, config, experiment_id, load_from_directory=None):
        """
        Initializes an experiment by initializing its variables, loading the corresponding dataset, loading the
        language information, and instantiating the model and the optimizer.

        :param config: The configurations of this experiment as a dictionary object.
        :param experiment_id: A unique identifier for this experiment.
        :param load_from_directory: A string indicating the path to a directory. If omitted, a new experiment is
        created. Else, an already serialized experiment will be read from the directory that is pointed to by this
        argument.
        """
        self.config = config
        self.id = experiment_id
        self.current_epoch = 0
        self.best_test_score = float('inf')
        self.best_epoch_number = -1
        self.loss_function = self.AutoPaddedLoss(torch.nn.CrossEntropyLoss(ignore_index=Alphabet.unknown_integer) )
        self.model = None
        self.optimizer = None
        
        # Find the corresponding dataset.
        assert config[consts.DATASET] in [consts.SIGMORPHON2020]
        if config[consts.DATASET] == consts.SIGMORPHON2020:
            # Create a data loader factory.
            data_loader_factory = SigmorphonData_Factory(config[consts.SIGMORPHON2020_ROOT],
                                                         config[consts.LANGUAGE_INFO_FILE])
            # Create a data loader for the training data.
            logging.getLogger(consts.MAIN).info('Creating the training dataset.')
            dataloader_kwargs = {'batch_size': self.config[consts.BATCH_SIZE], 'collate_type': 'unpacked'}
            self.train_loader = data_loader_factory.get_dataset(type=[consts.TRAIN],
                                                                families=self.config[consts.LANGUAGE_FAMILIES],
                                                                languages=self.config[consts.LANGUAGES],
                                                                dataloader_kwargs=dataloader_kwargs)
            # Create a data loader for the testing data.
            logging.getLogger(consts.MAIN).info('Creating the testing dataset.')
            self.test_loader = data_loader_factory.get_dataset(type=[consts.DEV],
                                                               families=self.config[consts.LANGUAGE_FAMILIES],
                                                               languages=self.config[consts.LANGUAGES],
                                                               dataloader_kwargs=dataloader_kwargs)
        else:
            raise Exception('Unsupported dataset.')

        # Instantiate the model indicated by the configurations.
        self.model = ModelFactory.create_model(config[consts.MODEL_ARCHITECTURE],
                                               self.config,
                                               self.train_loader.dataset.get_dimensionality()
                                               )
        
        # Move to the preferred device
        self.loss_function = self.loss_function.to(self.config[consts.DEVICE])
        self.model = self.model.to(self.config[consts.DEVICE])
        if self.config[consts.DATA_PARALLEL]:
            self.model = torch.nn.DataParallel(self.model)

        # Instantiate the optimizer indicated by the configurations.
        self.optimizer = OptimizerFactory.create_optimizer(config[consts.OPTIMIZER], self.model, self.config)

        # If the `load_from_directory` argument is given, load the state of the experiment from a file.
        if load_from_directory is not None:
            self.deserialize(load_from_directory)

    def run(self):
        """
        Executes the experiment and reports the results.

        :return: The highest score achieved by this experiment across the epochs.
        """
        # Initialize the WandB module to log the execution of this experiment.
        with wandb.init(project='TurkicWorkgroup', reinit=True,
                        name="Identifier: {} - Experiment: {:03d}".format(execution_identifier, self.id),
                        config=self.config, dir='../wandb', resume=self.config[consts.CONTINUE],
                        id="{}-{}".format(execution_identifier, self.id)):
            # Have WandB monitor the state of the model.
            wandb.watch(self.model)

            # At each epoch ...
            while self.current_epoch < self.config[consts.NUM_EPOCHS]:
                # Make a checkpoint every other `CHECKPOINT_STEP` epoch, starting with epoch 0.
                if self.current_epoch % self.config[consts.CHECKPOINT_STEP] == 0:
                    self.make_checkpoint()

                # Run an epoch of training.
                epoch_test_score = self.run_epoch()

                # Compare the results of the epoch to the best results so far and update the record if necessary.
                if epoch_test_score <= self.best_test_score:
                    self.best_test_score = epoch_test_score
                    self.best_epoch_number = self.current_epoch
                # Log the results
                wandb.log({'Best Training Score': self.best_test_score})
                self.current_epoch += 1

        # Report the best results across all epochs.
        return self.best_test_score

    def make_checkpoint(self):
        """
        Makes a checkpoint and persists the current state of the experiment to a file.
        """
        try:
            # Locate the checkpoint directory.
            checkpoint_dir = os.path.join(self.config[consts.EXPORT_DIR],
                                          "checkpoints",
                                          "experiment_%09d" % self.id,
                                          "epoch_%09d" % self.current_epoch)
            # If the checkpoint directory already exists, remove it.
            if os.path.isdir(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            # Create the checkpoint directory.
            os.makedirs(checkpoint_dir)

            # Persist the current state of the experiment.
            with torch.no_grad():
                self.serialize(checkpoint_dir)

            # Update the stamp indicating the newest checkpoint.
            epoch_stamp_path = os.path.join(self.config[consts.EXPORT_DIR], "epoch_stamp.pickle")
            with open(epoch_stamp_path, 'wb') as file:
                pickle.dump({consts.EXPERIMENT_ID: self.id, consts.EPOCH_NUMBER: self.current_epoch}, file)
        except IOError as exception:
            raise exception

    def serialize(self, directory):
        """
        Persist the state of the experiment into file.

        :param directory: A string pointing to the directory where the checkpoint is to be created.
        """
        # Store the language information.
        # with open(os.path.join(directory, "language_collection.pickle"), 'wb') as file:
        #     pickle.dump(self.language_collection, file)

        # Store the configuration of this experiment and the record of the best results so far.
        with open(os.path.join(directory, "dictionary.pickle"), 'wb') as file:
            pickle.dump({'config': self.config, 'id': self.id, 'current_epoch': self.current_epoch,
                         'best_test_score': self.best_test_score, 'best_epoch_number': self.best_epoch_number}, file)

        # Store the current state of the model.
        with open(os.path.join(directory, "model.pickle"), 'wb') as file:
            torch.save(self.model.state_dict(), file)

        # Store the current state of the optimizer.
        with open(os.path.join(directory, "optimizer.pickle"), 'wb') as file:
            torch.save(self.optimizer.state_dict(), file)

    def deserialize(self, directory):
        """
        Loads a previously serialized experiment from file.
        :param directory: A string pointing to the directory where the checkpoint is at.
        """
        # Load the language collection.
        # with open(os.path.join(directory, "language_collection.pickle"), 'rb') as file:
        #     self.language_collection = pickle.load(file)

        # Load the configuration of this experiment and the record of the best results so far.
        with open(os.path.join(directory, "dictionary.pickle"), 'rb') as file:
            dictionary = pickle.load(file)
            self.config = dictionary['config']
            self.id = dictionary['id']
            self.current_epoch = dictionary['current_epoch']
            self.best_test_score = dictionary['best_test_score']
            self.best_epoch_number = dictionary['best_epoch_number']

        # Load the state of the model.
        with open(os.path.join(directory, "model.pickle"), 'rb') as file:
            self.model.load_state_dict(torch.load(file))
            self.model = self.model.to(self.config[consts.DEVICE])

        # Load the state of the optimizer.
        with open(os.path.join(directory, "optimizer.pickle"), 'rb') as file:
            self.optimizer.load_state_dict(torch.load(file))

    def train(self):
        """
        Train the model for a single epoch.

        :return: The average training loss in this epoch, as a floating point number.
        """
        # Put the model in training mode.
        self.model.train()

        total_loss = 0.0
        
        
        batch_start_time = time.time()
        
        # For each batch of data ...
        for batch_idx, (input_batch, output_batch) in enumerate(self.train_loader):
            
            # Zero out the previous gradient information.
            self.optimizer.zero_grad()

            # Split the batch into semantic parts.
            family = input_batch.family
            language = input_batch.language
            tags = input_batch.tags
            lemma = input_batch.lemma
            form = output_batch.form
            tags_str = output_batch.tags_str
            lemma_str = output_batch.lemma_str
            form_str = output_batch.form_str
            
            #TODO: Unless the dataloader is sending the data to the appropriate device, maybe handle it here.
            
            # Run the model on this batch of data.
            probabilities = self.model(family, language, tags, lemma)
            outputs = [torch.argmax(probability, dim=1) for probability in probabilities]

            # Compute the batch loss.
            batch_loss = 0.0
            batch_size = len(tags)
            for i in range(batch_size):
                output_str = "".join([self.train_loader.dataset.alphabet_input[int(integral)]
                                      for integral in outputs[i]])
                language_family =\
                    self.train_loader.dataset.language_collection[int(family[i][0])]
                language_object = language_family[int(language[i][0])]
                logging.getLogger(consts.MAIN).debug(
                    "stem: {},"
                    "\ttarget: {},"
                    "\ttags: {}"
                    "\tlanguage: {}/{}"
                    "\toutput: '{}'".format(lemma_str[i], form_str[i], tags_str[i], language_family.name,
                                            language_object.name, output_str))
                batch_loss += self.loss_function(probabilities[i], form[i])

            # Update model parameter.
            batch_loss.backward()
            self.optimizer.step()
            
            
            #benchmark stuff
            batch_end_time = time.time()
            batches_per_second = 1.0/(batch_end_time-batch_start_time)
            batch_start_time = batch_end_time
            #benchmark Log the benchmark to wandb
            items_per_sec = int(len(output_batch.lemma_str))*batches_per_second
            wandb.log({"Batch Items/Sec":items_per_sec})

            # Log the outcome of this batch.
            total_loss += float(batch_loss)#clears the computation graph history.
            logging.getLogger(consts.MAIN
                ).info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tItems/s {:.2f}: '.format(
                                self.current_epoch, batch_idx * self.config[consts.BATCH_SIZE],
                                len(self.train_loader.dataset),
                                100. * batch_idx / len(self.train_loader),
                                batch_loss.item() / batch_size,
                                items_per_sec
                            )
                        )
            wandb.log({'Batch Training Loss': batch_loss})
            
            
            
        # Log and report the outcome of this epoch.
        wandb.log({'Epoch Training Loss': total_loss / len(self.train_loader)})
        return total_loss / len(self.train_loader)

    def test(self):
        """
        Test the model.

        :return: The test accuracy.
        """
        # Put the model in testing mode.
        self.model.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            # For each batch of data ...
            for batch_idx, (input_batch, output_batch) in enumerate(self.test_loader):
                # Split the batch into semantic parts.
                family = input_batch.family
                language = input_batch.language
                tags = input_batch.tags
                lemma = input_batch.lemma
                form = output_batch.form
                tags_str = output_batch.tags_str
                lemma_str = output_batch.lemma_str
                form_str = output_batch.form_str

                # Run the model on this batch of data.
                probabilities = self.model(family, language, tags, lemma)
                outputs = [torch.argmax(probability, dim=1) for probability in probabilities]

                # Compute the loss and the accuracy on this batch.
                batch_size = len(tags)
                for i in range(batch_size):
                    output_str = "".join([self.train_loader.dataset.alphabet_input[int(integral)]
                                          for integral in outputs[i]])
                    language_family =\
                        self.train_loader.dataset.language_collection[int(family[i][0])]
                    language_object = language_family[int(language[i][0])]
                    logging.getLogger(consts.MAIN).debug(
                        "stem: {},"
                        "\ttarget: {},"
                        "\ttags: {}"
                        "\tlanguage: {}/{}"
                        "\toutput: '{}'".format(lemma_str[i], form_str[i], tags_str[i], language_family.name,
                                                language_object.name, output_str))
                    # Keep track of loss and accuracy.
                    padding = torch.LongTensor([Alphabet.stop_integer] * (len(outputs[i]) - len(form[i])))
                    target = torch.cat([form[i], padding])
                    test_loss += self.loss_function(probabilities[i], target)
                    if target == output_str:
                        correct += 1

            # Compute and log the total loss and accuracy.
            test_loss /= len(self.test_loader.dataset)
            test_accuracy = correct / len(self.test_loader.dataset)
            logging.getLogger(consts.MAIN).info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                test_loss, correct, len(self.test_loader.dataset), 100. * test_accuracy))
            wandb.log({'Epoch Test Loss': test_loss, 'Epoch Test Accuracy': test_accuracy})

        # Report the accuracy.
        return test_accuracy

    def run_epoch(self):
        """
        Run the experiment for one epoch and report its test accuracy. It trains the model for one epoch and then
        tests it.

        :return: The test accuracy at the end of this epoch, as a floating point number.
        """
        self.train()
        return self.test()
