import consts
from data import dataset
from data.sigmorphon2020_data_reader import *
import logging
from models.seq2seq import Seq2Seq
import os
import pickle
import shutil
import sys
import time
import torch
import wandb


execution_identifier = str(time.time())


class Experiment:
    def __init__(self, config=None, experiment_id=-1, load_from_directory=None):
        self.config = config
        self.id = experiment_id
        self.current_epoch = 0
        self.best_test_score = float('inf')
        self.best_epoch_number = -1
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=dataset.PADDING_TOKEN)
        self.language_collection = pickle.load(config[consts.LANGUAGE_INFO_FILE])
        if config[consts.DATASET] == consts.SIGMORPHON2020:
            self.train_loader = create_data_loader_from_sigmorphon2020(self.config, is_train=True)
            self.test_loader = create_data_loader_from_sigmorphon2020(self.config, is_train=False)
        if config[consts.MODEL_ARCHITECTURE] == consts.SEQ2SEQ:
            self.model = Seq2Seq(self.train_loader.alphabet_vector_dim, self.train_loader.tags_vector_dim)
        else:
            print('Bad argument: --model-architecture {} is not recognized.'.format(
                config[consts.MODEL_ARCHITECTURE]),
                file=sys.stderr)
            exit(64)
        if config[consts.OPTIMIZER] == consts.ADADELTA:
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=config[consts.LR])
        else:
            print('Bad argument: --optimizer {} is not recognized.'.format(config[consts.OPTIMIZER]),
                  file=sys.stderr)
            exit(64)

        if load_from_directory is not None:
            self.deserialize(load_from_directory)

    def run(self):
        with wandb.init(project='TurkicWorkgroup', reinit=True,
                        name="Identifier: {} - Experiment: {:03d}".format(execution_identifier, self.id),
                        config=self.config, dir='../wandb', resume=self.config[consts.CONTINUE],
                        id="{}-{}".format(execution_identifier, self.id)):
            wandb.watch(self.model)
            while self.current_epoch < self.config[consts.NUM_EPOCHS]:
                if self.current_epoch % self.config[consts.CHECKPOINT_STEP] == 0:
                    self.make_checkpoint()
                epoch_test_score = self.run_epoch()
                if epoch_test_score <= self.best_test_score:
                    self.best_test_score = epoch_test_score
                    self.best_epoch_number = self.current_epoch
                wandb.log({'Best Training Score': self.best_test_score})
                self.current_epoch += 1
        return self.best_test_score

    def make_checkpoint(self):
        try:
            checkpoint_dir = os.path.join(self.config[consts.EXPORT_DIR],
                                          "checkpoints",
                                          "experiment_%09d" % self.id,
                                          "epoch_%09d" % self.current_epoch)
            if os.path.isdir(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir)
            with torch.no_grad():
                self.serialize(checkpoint_dir)
            epoch_stamp_path = os.path.join(self.config[consts.EXPORT_DIR], "epoch_stamp.pickle")
            with open(epoch_stamp_path, 'wb') as file:
                pickle.dump({consts.EXPERIMENT_ID: self.id, consts.EPOCH_NUMBER: self.current_epoch}, file)
        except IOError as exception:
            raise exception

    def serialize(self, directory):
        pickle.dump(self.language_collection, os.path.join(directory, "language_collection.pickle"))
        with open(os.path.join(directory, "dictionary.pickle"), 'wb') as file:
            pickle.dump({'config': self.config, 'id': self.id, 'current_epoch': self.current_epoch,
                         'best_test_score': self.best_test_score, 'best_epoch_number': self.best_epoch_number}, file)
        with open(os.path.join(directory, "model.pickle"), 'wb') as file:
            torch.save(self.model.state_dict(), file)
        with open(os.path.join(directory, "optimizer.pickle"), 'wb') as file:
            torch.save(self.optimizer.state_dict(), file)

    def deserialize(self, directory):
        self.language_collection = pickle.load(os.path.join(directory, "language_collection.pickle"))
        with open(os.path.join(directory, "dictionary.pickle"), 'rb') as file:
            dictionary = pickle.load(file)
            self.config = dictionary['config']
            self.id = dictionary['id']
            self.current_epoch = dictionary['current_epoch']
            self.best_test_score = dictionary['best_test_score']
            self.best_epoch_number = dictionary['best_epoch_number']
        with open(os.path.join(directory, "model.pickle"), 'rb') as file:
            self.model.load_state_dict(torch.load(file))
            self.model = self.model.to(self.config[consts.DEVICE])
        with open(os.path.join(directory, "optimizer.pickle"), 'rb') as file:
            self.optimizer.load_state_dict(torch.load(file))

    def train(self):
        self.model.train()
        total_loss = 0.0
        for batch_idx, (families, languages, tags, lemmata, forms, tags_strs, lemmata_strs, forms_strs) in\
                enumerate(self.train_loader):
            batch_size = len(families)
            self.optimizer.zero_grad()
            probabilities, outputs = self.model(families, languages, tags, lemmata)
            batch_loss = 0.0
            for i in range(batch_size):
                logging.getLogger(consts.MAIN).debug(
                    "stem: {},\ttarget: {},\ttags: {}\tlanguage: {}/{}"
                    "\noutput: {},".format(lemmata_strs[i], forms_strs[i], tags_strs[i], families[i], languages[i],
                                           outputs[i]))
                batch_loss += self.loss_function(probabilities[i], forms[i])
            batch_loss /= batch_size
            total_loss += batch_loss
            batch_loss.backward()
            self.optimizer.step()
            logging.getLogger(consts.MAIN).info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                self.current_epoch, batch_idx * self.config[consts.BATCH_SIZE], len(self.train_loader.dataset),
                100. * batch_idx / len(self.train_loader), batch_loss.item()))
            wandb.log({'Batch Training Loss': batch_loss})
        wandb.log({'Epoch Training Loss': total_loss / len(self.train_loader)})
        return total_loss / len(self.train_loader)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (families, languages, tags, lemmata, forms, tags_strs, lemmata_strs, forms_strs) in\
                    enumerate(self.test_loader):
                batch_size = len(lemmata)
                probabilities, outputs = self.model(families, languages, tags, lemmata)
                for i in range(batch_size):
                    logging.getLogger(consts.MAIN).debug(
                        "stem: {},\ttarget: {},\ttags: {}\tlanguage: {}/{}"
                        "\noutput: {},".format(lemmata_strs[i], forms_strs[i], tags_strs[i], families[i], languages[i],
                                               outputs[i]))
                    test_loss += self.loss_function(probabilities[i], forms[i])

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
            test_loss /= len(self.test_loader.dataset)
            test_accuracy = correct / len(self.test_loader.dataset)
            logging.getLogger(consts.MAIN).info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                test_loss, correct, len(self.test_loader.dataset), 100. * test_accuracy))
            wandb.log({'Epoch Test Loss': test_loss, 'Epoch Test Accuracy': test_accuracy})
        return test_accuracy

    def run_epoch(self):
        self.train()
        return self.test()
