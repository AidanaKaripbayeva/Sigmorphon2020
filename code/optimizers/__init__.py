import argparse
import torch


class OptimizerFactory:
    """
    Factory class in charge of creating optimizers and their corresponding argument parsers.
    """
    optimizers = ['adadelta']

    @staticmethod
    def create_optimizer(optimizer_name, model, config):
        """
        Given the name of the optimizer and its corresponding arguments, instantiates and returns an optimizer.

        :param optimizer_name: Optimizer name as a string.
        :param model: The model object that the optimizer will train.
        :param config: The command-line arguments, as a dictionary object.
        :return: The instantiated optimizer.
        """
        if optimizer_name == 'adadelta':
            return torch.optim.Adadelta(model.parameters(),
                                        lr=config['adadelta_lr'],
                                        rho=config['adadelta_rho'],
                                        weight_decay=config['adadelta_weight_decay'],
                                        eps=config['adadelta_eps'])
        else:
            raise Exception('Optimizer \'{}\' not supported.'.format(optimizer_name))

    @staticmethod
    def get_parser(optimizer_name):
        """
        Instantiates an ArgumentParser for the particular optimizer given in the input.

        :param optimizer_name: Name of the optimizer as a string.
        :return: The instantiated ArgumentParser object.
        """
        parser = argparse.ArgumentParser(add_help=False)

        if optimizer_name == 'adadelta':
            parser.add_argument('--adadelta-eps', type=float, default=1e-10,
                                help='Adadelta: term added to the denominator to improve numerical stability')
            parser.add_argument('--adadelta-lr', type=float, default=1e-2, help='Adadelta: learning rate.')
            parser.add_argument('--adadelta-rho', type=float, default=0.9,
                                help='Adadelta: coefficient used for computing a running average of squared'
                                     'gradients.')
            parser.add_argument('--adadelta-weight-decay', type=float, default=0,
                                help='Adadelta: weight decay factor.')
        else:
            raise Exception('Optimizer \'{}\' not supported.'.format(optimizer_name))

        return parser

    @staticmethod
    def get_all_parsers():
        """Get a list of all argument parsers for different model architectures."""
        return [OptimizerFactory.get_parser(optimizer) for optimizer in OptimizerFactory.optimizers]
