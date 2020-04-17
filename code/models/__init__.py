from .dumbcopy import DumbCopy
from .dumbertransducer import DumberTransducer
from .seq2seq import Seq2Seq
from .babytransducer import BabyTransducer

import argparse
import consts


class ModelFactory:
    """
    Factory class in charge of creating models and their corresponding argument parsers.
    """
    architectures = ['seq2seq', "dummy", "dumber", 'baby-transducer']

    @staticmethod
    def create_model(architecture_name, config, dimensionality):
        """
        Given the name of the model architecture and its corresponding arguments, instantiates and returns a model.

        :param architecture_name: Architecture name as a string.
        :param config: The command-line arguments, as a dictionary object.
        :param dimensionality: A dictionary object informing the model of the dimensions of the different input parts.
        :return: The instantiated model.
        """
        if architecture_name == 'seq2seq':
            return Seq2Seq(dimensionality[consts.INPUT_SYMBOLS],
                           dimensionality[consts.NUM_LANGUAGES],
                           dimensionality[consts.TAGS],
                           embedding_dim=int(config['seq2seq_embedding_dimension']),
                           hidden_dim=config['seq2seq_hidden_dimension'],
                           num_layers=config['seq2seq_num_layers'])
        elif architecture_name == 'dummy':
            return DumbCopy(dimensionality[consts.INPUT_SYMBOLS],
                           dimensionality[consts.NUM_LANGUAGES],
                           dimensionality[consts.TAGS],
                           embedding_dim=int(config['dummy_embedding_dimension']),
                           hidden_dim=config['dummy_hidden_dimension'],
                           num_layers=config['dummy_num_layers'])
        elif architecture_name == 'dumber':
            return DumberTransducer(dimensionality[consts.INPUT_SYMBOLS],
                           dimensionality[consts.NUM_LANGUAGES],
                           dimensionality[consts.TAGS],
                           embedding_dim=int(config['dumber_embedding_dimension']),
                           hidden_dim=config['dumber_hidden_dimension'],
                           num_layers=config['dumber_num_layers'])
        elif architecture_name == 'baby-transducer':
            return BabyTransducer(alphabet_size=dimensionality[consts.INPUT_SYMBOLS],
                                  tag_vector_dim=dimensionality[consts.TAGS][0],
                                  embed_dim=config['baby_transducer_embedding_dimension'],
                                  src_hid_size=config['baby_transducer_source_hidden_size'],
                                  src_nb_layers=config['baby_transducer_source_num_layers'],
                                  trg_hid_size=config['baby_transducer_target_hidden_size'],
                                  trg_nb_layers=config['baby_transducer_target_num_layers'],
                                  attention_dim=config['baby_transducer_attention_dim'],
                                  dropout_p=config['baby_transducer_dropout_probability'],
                                  teacher_force=config['baby_transducer_teacher_force'])
        else:
            raise Exception('Architecture \'{}\' not supported.'.format(architecture_name))

    @staticmethod
    def get_parser(architecture_name):
        """
        Instantiates an ArgumentParser for the particular model architecture given in the input.

        :param architecture_name: Name of the architecture as a string.
        :return: The instantiated ArgumentParser object.
        """
        parser = argparse.ArgumentParser(add_help=False)

        if architecture_name == 'seq2seq':
            parser.add_argument('--seq2seq-embedding-dimension', type=int, default=40,
                                help='The dimension of the output of the embedding layer.')
            parser.add_argument('--seq2seq-hidden-dimension', type=int, default=10,
                                help='The dimension of the hidden layer in the LSTM unit in the encoder.')
            parser.add_argument('--seq2seq-num-layers', type=int, default=3,
                                help='The number of layers in the LSTM units in the encoder and in the'
                                     'decoder.')
        elif architecture_name == 'dummy':
            parser.add_argument('--dummy-embedding-dimension', type=int, default=40,
                                help='The dimension of the output of the embedding layer.')
            parser.add_argument('--dummy-hidden-dimension', type=int, default=10,
                                help='The dimension of the hidden layer in the LSTM unit in the encoder.')
            parser.add_argument('--dummy-num-layers', type=int, default=3,
                                help='The number of layers in the LSTM units in the encoder and in the'
                                     'decoder.')
        elif architecture_name == 'dumber':
            parser.add_argument('--dumber-embedding-dimension', type=int, default=40,
                                help='The dimension of the output of the embedding layer.')
            parser.add_argument('--dumber-hidden-dimension', type=int, default=10,
                                help='The dimension of the hidden layer in the LSTM unit in the encoder.')
            parser.add_argument('--dumber-num-layers', type=int, default=1,
                                help='The number of layers in the LSTM units in the encoder and in the'
                                     'decoder.')
        elif architecture_name == 'baby-transducer':
            parser.add_argument('--baby-transducer-embedding-dimension', type=int, default=100,
                                help='The dimension of the output of the embedding layer.')
            parser.add_argument('--baby-transducer-source-hidden-size', type=int, default=200)
            parser.add_argument('--baby-transducer-source-num-layers', type=int, default=1)
            parser.add_argument('--baby-transducer-target-hidden-size', type=int, default=200)
            parser.add_argument('--baby-transducer-target-num-layers', type=int, default=1)
            parser.add_argument('--baby-transducer-attention-dim', type=int, default=100)
            parser.add_argument('--baby-transducer-dropout-probability', type=float, default=0.1,
                                help='Drop out probability.')
            parser.add_argument('--baby-transducer-teacher-force', type=float, default=0.9,
                                help='The probability with which to force the target into the sequence while'
                                     'decoding.')
        else:
            raise Exception('Architecture \'{}\' not supported.'.format(architecture_name))

        return parser

    @staticmethod
    def get_all_parsers():
        """Get a list of all argument parsers for different model architectures."""
        return [ModelFactory.get_parser(architecture) for architecture in ModelFactory.architectures]
