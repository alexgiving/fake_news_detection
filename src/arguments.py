from argparse import ArgumentParser, Namespace


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
        action='store_true'
        )

    return parser.parse_args()
