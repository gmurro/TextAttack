from argparse import ArgumentParser

from textattack.commands import TextAttackCommand
class TrainModelCommand(TextAttackCommand):
    """
    The TextAttack train module:
    
        A command line parser to train a model from user specifications.
    """
    
    def run(self):
        raise NotImplementedError('cant benchmark yet')

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser("train", help="train a model")