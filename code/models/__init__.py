from .seq2seq import Seq2Seq
import argparse
import consts


class ModelFactory:
    """
    Factory class in charge of creating models and their corresponding argument parsers.
    """
    architectures = ['seq2seq']

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
                                help='The number of layers in the LSTM units in the encoder and in the decoder.')
        else:
            raise Exception('Architecture \'{}\' not supported.'.format(architecture_name))

        return parser

    @staticmethod
    def get_all_parsers():
        """Get a list of all argument parsers for different model architectures."""
        return [ModelFactory.get_parser(architecture) for architecture in ModelFactory.architectures]
