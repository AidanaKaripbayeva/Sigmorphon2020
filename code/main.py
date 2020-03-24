import argparse
import consts
import experiments.experiment
from experiments.experiment import Experiment
import logging
import os
import pickle
import sys
import torch


def get_parser():
    """
    Constructs and returns a parser object to handle the command-line arguments.

    :return: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()

    # Generic options
    parser.add_argument('--checkpoint-step', type=int, default=1,
                        help='Number of epochs between successive checkpoint creations')
    parser.add_argument('--config-file', type=str, default=[], nargs='*',
                        help='File(s) to read the command-line arguments from')
    parser.add_argument('--continue', action='store_true',
                        help='Continue the execution of the last experiment saved into the export directory')
    parser.add_argument('--debug', action='store_true', help='Show debug messages')
    parser.add_argument('--export-dir', type=str, required=True, help='Export directory')
    parser.add_argument('--no-gpu', action='store_true', help='Use CPU')

    # Data options
    parser.add_argument('--batch-size', type=int, default=[16], nargs='*', help='Batch size(s)')
    parser.add_argument('--dataset', type=str, default=[consts.SIGMORPHON2020], nargs='*',
                        choices=[consts.SIGMORPHON2020], help='Dataset(s) to train on')
    parser.add_argument('--sigmorphon2020-root', type=str, help='Root directory for the SIGMORPHON 2020 dataset')

    # Language options
    parser.add_argument('--language-families', type=str, nargs='*', default=[],
                        help='The families of languages to load the data for.'
                             ' If not provided, all available families will be used.')
    parser.add_argument('--language-info-file', type=str, required=True, help='The language information file.')
    parser.add_argument('--languages', type=str, nargs='*', default=[],
                        help='The languages to load the data for.'
                             ' If not provided, all available languages will be used.')
    parser.add_argument('--no-read-language-info-from-data', action='store_true',
                        help='Don\'t read the data from the data and store it in the location give by '
                             '--language-info-dir. '
                             ' If this flag not present, language information is read from the directory given by'
                             ' --language-info-dir.')

    # Optimizer options
    parser.add_argument('--optimizer', type=str, default=[consts.ADADELTA], choices=[consts.ADADELTA], nargs='*',
                        help='Optimizer algorithm(s)')
    parser.add_argument('--lr', type=float, default=[0.1], nargs='*', help='Learning rate(s)')
    parser.add_argument('--lr-decay', type=float, default=[0.1], nargs='*', help='Learning rate decay factor(s)')
    parser.add_argument('--lr-step-size', type=int, default=[20], nargs='*',
                        help='Number(s) of steps between successive learning rate decays')
    parser.add_argument('--num-epochs', type=int, default=30, help='Number(s) of epochs')

    # Model options
    parser.add_argument('--model-architecture', type=str, default=[consts.SEQ2SEQ], nargs='*',
                        choices=[consts.SEQ2SEQ], help='Model architecture(s)')

    return parser


def get_options(parser=None):
    """
    Retrieves and parses the command-line argument. It then checks the arguments for errors. If the argument for
    continuing a previous execution is passed, then all the other command-line arguments are ignored and instead the
     arguments for the previous execution are loaded from file.

    :param parser: The parser to use to parse the command-line arguments. If omittedd, the default parser given by
    get_parser() is used.
    :return: A dictionary containing the parsed arguments.
    """
    if parser is None:
        parser = get_parser()
    inline_options = vars(parser.parse_args())
    inline_options[consts.DEVICE] =\
        torch.device(consts.CUDA if torch.cuda.is_available() and not inline_options[consts.NO_GPU] else consts.CPU)

    # Load inline_options
    # Check if continue option is given. If so, replace the command-line arguments with those loaded from a file.
    if inline_options[consts.CONTINUE]:
        try:
            options_path = os.path.join(inline_options[consts.EXPORT_DIR], 'options.pickle')
            with open(options_path, 'rb') as file:
                unpickled_options = pickle.load(file)
            for option, value in unpickled_options.items():
                inline_options[option] = value
            inline_options[consts.CONTINUE] = True
        except FileNotFoundError as _:
            print('Invalid load directory.', file=sys.stderr)
            exit(66)

    # Load the options given through a configuration file.
    options_array = [inline_options.copy()]
    for config_file in inline_options[consts.CONFIG_FILE]:
        try:
            with open(config_file, 'r') as file:
                options_array.append({**inline_options, **vars(parser.parse_args(file.read()))})
        except IOError as _:
            print('Bad argument: --config-file {}.'.format(config_file), file=sys.stderr)
            exit(64)

    # Check options for errors
    for options in options_array:
        if options[consts.CHECKPOINT_STEP] < 0:
            print('Bad argument: --checkpoint-step must be non-negative.', file=sys.stderr)
            exit(64)
        if options[consts.EXPORT_DIR] is None:
            print('Missing argument: --export-dir', file=sys.stderr)
            exit(66)
        if options[consts.NO_GPU] is None:
            options[consts.NO_GPU] = False
        for batch_size in options[consts.BATCH_SIZE]:
            if batch_size <= 0:
                print('Bad argument: --batch-size must be non-negative.', file=sys.stderr)
                exit(64)
        if len(options[consts.DATASET]) == 0:
            print('Missing argument: --dataset', file=sys.stderr)
            exit(66)
        for dataset in options[consts.DATASET]:
            if dataset == consts.SIGMORPHON2020:
                if options[consts.SIGMORPHON2020_ROOT] is None:
                    print('Missing argument: --sigmorphon2020-root', file=sys.stderr)
                    exit(66)
            else:
                print('Bad argument: --dataset {} is not recognized.'.format(options[consts.DATASET]), file=sys.stderr)
                exit(64)
        for optimizer in options[consts.OPTIMIZER]:
            if optimizer == consts.ADADELTA:
                if len(options[consts.LR]) == 0:
                    print('Missing argument: --lr.', file=sys.stderr)
                    exit(64)
            else:
                print('Bad argument: --optimizer {} not recognized.'.format(optimizer), file=sys.stderr)
                exit(64)
        for lr in options[consts.LR]:
            if lr <= 0:
                print('Bad argument: --lr must be non-negative.', file=sys.stderr)
                exit(64)
        for lr_decay in options[consts.LR_DECAY]:
            if not 0 < lr_decay < 1:
                print('Bad argument: --lr-decay must be in (0,1).', file=sys.stderr)
                exit(64)
        for lr_step_size in options[consts.LR_STEP_SIZE]:
            if lr_step_size <= 0:
                print('Bad argument: --lr-step-size must be non-negative.', file=sys.stderr)
                exit(64)
        if options[consts.NUM_EPOCHS] <= 0:
            print('Bad argument: --num-epochs must be positive.', file=sys.stderr)
            exit(64)

    return inline_options


def iterate_model_architecture_configs(options):
    """
    If multiple models are chosen, creates a separate configuration dictionary for each choice of the model.

    :param options: A dictionary object containing the parsed command-line arguments.
    :return: A dictionary object for each choice of the model architecture, containing the options specific to that
    choice.
    """
    for model_architecture in options[consts.MODEL_ARCHITECTURE]:
        config = options.copy()
        config[consts.MODEL_ARCHITECTURE] = model_architecture
        yield config


def iterate_dataset_configs(options):
    """
    If multiple datasets are chosen, creates a separate configuration dictionary for each choice of the dataset.

    :param options: A dictionary object containing the parsed command-line arguments.
    :return: A dictionary object for each choice of the dataset, containing the options specific to that choice.
    """
    for dataset in options[consts.DATASET]:
        config = options.copy()
        config[consts.DATASET] = dataset
        yield config


def iterate_optimizer_configs(options):
    """
    If multiple optimizers are chosen, creates a separate configuration dictionary for each choice of the optimizer.

    :param options: A dictionary object containing the parsed command-line arguments.
    :return: A dictionary object for each choice of the model architecture, containing the options specific to that choice.
    """
    for batch_size in options[consts.BATCH_SIZE]:
        for optimizer in options[consts.OPTIMIZER]:
            if optimizer == consts.ADADELTA:
                for lr in options[consts.LR]:
                    for lr_decay in options[consts.LR_DECAY]:
                        for lr_step_size in options[consts.LR_STEP_SIZE]:
                            config = options.copy()
                            config[consts.BATCH_SIZE] = batch_size
                            config[consts.OPTIMIZER] = optimizer
                            config[consts.LR] = lr
                            config[consts.LR_DECAY] = lr_decay
                            config[consts.LR_STEP_SIZE] = lr_step_size
                            yield config
            else:
                print('Bad argument: --optimizer {} not recognized.'.format(optimizer), file=sys.stderr)
                exit(64)


def iterate_configs(parser=None, options=None):
    """
    Takes the command-line arguments and iterates through the different configurations of hyperparameters given
    through the command-line arguments.

    :param parser: The parser to use to parse the command-line arguments. If omitted, the default parser given by
    get_parser() is used.
    :param options: A dictionary object containing the command-line arguemnts. If omitted, the arguments are
    retrieved from the command-line.
    :return:
    """
    if parser is None:
        parser = get_parser()
    if options is None:
        options = get_options()

    # Make a list of configuration files referenced in the command-line arguments.
    options_array = [{**options, consts.CONFIG_FILE: None}]
    for config_file in options[consts.CONFIG_FILE]:
        try:
            with open(config_file, 'r') as file:
                options_array.append(
                    {**options, **vars(parser.parse_args(file.read())), consts.CONFIG_FILE: config_file})
        except IOError as _:
            print('Bad argument: --config-file {}.'.format(config_file), file=sys.stderr)
            exit(64)

    # Walk through individual configuration files.
    for options in options_array:
        config1 = options.copy()
        # Walk through different configurations of hyperparameters.
        for config2 in iterate_model_architecture_configs(config1):
            for config3 in iterate_dataset_configs(config2):
                for config in iterate_optimizer_configs(config3):
                    yield config


def main():
    """
    Main function for a training task.
    """
    parser = get_parser()
    options = get_options(parser)

    # Set up the logger.
    logger = logging.getLogger(consts.MAIN)
    logger.setLevel(logging.DEBUG if options[consts.DEBUG] else logging.INFO)
    file_handler = logging.FileHandler(os.path.join(options[consts.EXPORT_DIR], 'log.txt'), mode='w')
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    # Log the options given through the command-line arguments.
    logger.info('options: {}'.format(str(options)))

    experiment_id = 0
    status_path = os.path.join(options[consts.EXPORT_DIR], "status.pickle")
    # Check if the execution is a new one or a resumption of a previous experiment.
    if not options[consts.CONTINUE]:
        # Set up a new execution.
        options_path = os.path.join(options[consts.EXPORT_DIR], 'options.pickle')
        with open(options_path, 'wb') as file:
            pickle.dump(options, file)
        best_experiment_test_score = -float('inf')
        best_experiment_id = -1
        best_epoch_num = -1
        best_config = None
        status = 'working'
        with open(status_path, 'wb') as file:
            pickle.dump([best_experiment_test_score, best_experiment_id, best_config, status], file)
        with open(os.path.join(options[consts.EXPORT_DIR], 'id'), 'w') as file:
            file.write(experiments.experiment.execution_identifier)
    else:
        # Load the old execution from the export directory.
        epoch_stamp_path = os.path.join(options[consts.EXPORT_DIR], "epoch_stamp.pickle")
        with open(epoch_stamp_path, 'rb') as file:
            dictionary = pickle.load(file)
        with open(status_path, 'rb') as file:
            best_experiment_test_score, best_experiment_id, best_epoch_num, best_config, status = pickle.load(file)
        with open(os.path.join(options[consts.EXPORT_DIR], 'id'), 'r') as file:
            experiments.experiment.execution_identifier = file.read()

    # Check if the execution is still in progress. This check should fail when an ended execution is resumed.
    if status == 'working':
        # Iterate through the different configurations of hyperparameters ad create an experiment for each.
        for config in iterate_configs(parser, options):
            # If this a resumed execution, check if this experiment has already had finished.
            if options[consts.CONTINUE] and experiment_id < dictionary[consts.EXPERIMENT_ID]:
                experiment_id += 1
                continue
            # If this a resumed execution and this is the experiment that was running when the last checkpoint was
            # created.
            elif options[consts.CONTINUE] and experiment_id == dictionary[consts.EXPERIMENT_ID]:
                # Log the configurations of the present experiment.
                logger.info('continuing on config: {}'.format(str(config)))
                checkpoint_dir = os.path.join(config[consts.EXPORT_DIR],
                                              "checkpoints",
                                              "experiment_%09d" % experiment_id,
                                              "epoch_%09d" % dictionary[consts.EPOCH_NUMBER])
                # Create an experiment for the configuration at hand.
                experiment = Experiment(config=config, experiment_id=experiment_id, load_from_directory=checkpoint_dir)
            # If this is a new experiment.
            else:
                logger.info('starting on config: {}'.format(str(config)))
                # Create an experiment for the configuration at hand.
                experiment = Experiment(config=config, experiment_id=experiment_id)
            # Run the present experiment.
            experiment_test_score = experiment.run()
            # Record the results of the experiment and compare them to the results so far.
            logger.info('Experiment {} test score: {}'.format(experiment_id, experiment_test_score))
            if experiment_test_score > best_experiment_test_score:
                best_experiment_test_score = experiment_test_score
                best_experiment_id = experiment_id
                best_epoch_num = experiment.best_epoch_number
                best_config = config
            # Store the best results so far in a file.
            with open(status_path, 'wb') as file:
                pickle.dump([best_experiment_test_score, best_experiment_id, best_epoch_num, best_config, status],
                            file)
            experiment_id += 1
        # Mark the execution as over.
        status = 'ended'
        # Store the best results in a file.
        with open(status_path, 'wb') as file:
            pickle.dump([best_experiment_test_score, best_experiment_id, best_epoch_num, best_config, status], file)
    # Report the best results.
    logger.info('Execution is over. Best experiment test score: {}'
                '\nBest experiment config: {}'.format(best_experiment_test_score, str(best_config)))


if __name__ == "__main__":
    main()
