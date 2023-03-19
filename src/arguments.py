from argparse import ArgumentParser, Namespace

from src.model import ModelEnum
from src.optimizer import OptimizerEnum


def parse() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        default='cpu'
        )

    parser.add_argument(
        "--fake-path",
        type=str,
        required=True
        )

    parser.add_argument(
        "--true-path",
        type=str,
        required=True
        )

    parser.add_argument(
        "--cache-folder",
        type=str,
        default='./cache/'
        )

    parser.add_argument(
        "--batch-size",
        type=int
        )

    parser.add_argument(
        "--epoches",
        type=int
        )

    parser.add_argument(
        "--is-half",
        action='store_true',
        help='Used to train model at half presision'
        )
    
    parser.add_argument(
        "--last-states",
        type=int,
        default=1,
        help='Define how many last states from embedding are used for training'
        )
    
    parser.add_argument(
        "--arch",
        type=ModelEnum,
        required=True,
        choices=list(ModelEnum),
        help='Define model to train'
        )

    parser.add_argument(
        "--optim",
        type=OptimizerEnum,
        required=True,
        choices=list(OptimizerEnum),
        help='Define optimizer to train'
        )

    return parser.parse_args()
