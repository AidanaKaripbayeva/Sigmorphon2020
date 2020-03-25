from data import dataset
from data.sigmorphon2020_data_reader import *
import logging
from models import ModelFactory
from optimizers import OptimizerFactory
import os
import pickle
import shutil
import sys
import time
import torch
import wandb

# Mark the start of the execution. This is used to generate a meaningful name for the execution.
execution_identifier = str(time.time())


class Experiment:
    """
    The class steering the flow of an experiment, persisting its intermediary results and reporting its final result.
    """

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
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=Alphabet.padding_integer)

        if True:  # TODO: fix up this block after the new data loader interface arrives.
            # Find the corresponding dataset.
            assert config[consts.DATASET] in [consts.SIGMORPHON2020]
            if config[consts.DATASET] == consts.SIGMORPHON2020:
                # Create a data loader for the training data.
                logging.getLogger(consts.MAIN).info('Creating the training dataset.')
                self.train_loader, tag_dimension = create_data_loader_from_sigmorphon2020(self.config, is_train=True)
                # Create a data loader for the testing data.
                logging.getLogger(consts.MAIN).info('Creating the testing dataset.')
                self.test_loader, _ = create_data_loader_from_sigmorphon2020(self.config, is_train=False)
            else:
                raise Exception('Unsupported dataset.')

            # Check if the user wants to construct language information from file or if they want to load it from a
            # previously serialized file.
            if config[consts.NO_READ_LANGUAGE_INFO_FROM_DATA] or\
                    not os.path.exists(config[consts.LANGUAGE_INFO_FILE]):
                # Process the data set to construct the language information.
                try:
                    logging.getLogger(consts.MAIN).info('Processing the dataset for language information.')
                    if consts.SIGMORPHON2020 in config[consts.DATASET]:
                        self.language_collection = \
                            compile_language_collection_from_sigmorphon2020(config[consts.SIGMORPHON2020_ROOT])
                        # Store the resulting information for future use in other experiments.
                        with open(config[consts.LANGUAGE_INFO_FILE], 'wb+') as file:
                            pickle.dump(self.language_collection, file)
                        # Mark a flag so the future experiments of the current execution won't process for language
                        # information again.
                        config[consts.NO_READ_LANGUAGE_INFO_FROM_DATA] = False
                    else:
                        raise Exception('Unsupported dataset.')
                except FileNotFoundError as _:
                    print('Invalid language file.', file=sys.stderr)
                    exit(66)
            # If the user wants to load the language information from a previously serialized file.
            else:
                logging.getLogger(consts.MAIN).info('Loading language information from ' +
                                                    config[consts.LANGUAGE_INFO_FILE] + ".")
                with open(config[consts.LANGUAGE_INFO_FILE], 'rb') as file:
                    self.language_collection = pickle.load(file)

            self.train_loader.language_collection = self.language_collection
            self.train_loader.tag_dimension = 356

        # Instantiate the model indicated by the configurations.
        self.model = ModelFactory.create_model(config[consts.MODEL_ARCHITECTURE], self.train_loader, self.config)

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
        with open(os.path.join(directory, "language_collection.pickle"), 'wb') as file:
            pickle.dump(self.language_collection, file)

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
        with open(os.path.join(directory, "language_collection.pickle"), 'rb') as file:
            self.language_collection = pickle.load(file)

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
        # For each batch of data ...
        for batch_idx, (families, languages, tags, lemmata, forms, tags_strs, lemmata_strs, forms_strs) in\
                enumerate(self.train_loader):
            # Zero out the previous gradient information.
            self.optimizer.zero_grad()

            # Run the model on this batch of data.
            probabilities, outputs = self.model(families, languages, tags, lemmata)

            # Compute the batch loss.
            batch_loss = 0.0
            batch_size = len(families)
            for i in range(batch_size):
                logging.getLogger(consts.MAIN).debug(
                    "stem: {},\ttarget: {},\ttags: {}\tlanguage: {}/{}"
                    "\noutput: {},".format(lemmata_strs[i], forms_strs[i], tags_strs[i], families[i], languages[i],
                                           outputs[i]))#TODO print output in textual form
                batch_loss += self.loss_function(probabilities[i], forms[i])

            # Update model parameter.
            batch_loss.backward()
            self.optimizer.step()

            # Log the outcome of this batch.
            total_loss += batch_loss
            logging.getLogger(consts.MAIN).info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                self.current_epoch, batch_idx * self.config[consts.BATCH_SIZE], len(self.train_loader.dataset),
                100. * batch_idx / len(self.train_loader), batch_loss.item() / batch_size))
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
            for batch_idx, (families, languages, tags, lemmata, forms, tags_strs, lemmata_strs, forms_strs) in\
                    enumerate(self.test_loader):
                # Run the model on this batch of data.
                probabilities, outputs = self.model(families, languages, tags, lemmata)

                # Compute the loss and the accuracy on this batch.
                batch_size = len(lemmata)
                for i in range(batch_size):
                    logging.getLogger(consts.MAIN).debug(
                        "stem: {},\ttarget: {},\ttags: {}\tlanguage: {}/{}"
                        "\noutput: {},".format(lemmata_strs[i], forms_strs[i], tags_strs[i], families[i],
                                               languages[i], outputs[i]))
                    # Aggregate the loss.
                    test_loss += self.loss_function(probabilities[i], forms[i])
                    # Check if the answer was correct.
                    correct += 1
                    for j in range(len(outputs[i])):
                        if j < len(forms[i]):
                            if outputs[i][j] != forms[i][j]:
                                correct -= 1
                                break
                        else:
                            if outputs[i][j] != dataset.PADDING_TOKEN:
                                correct -= 1
                                break

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
