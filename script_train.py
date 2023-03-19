from pathlib import Path

import torch

from src.arguments import parse
from src.dataset import FakeNewsDataset
from src.model import ModelEnum, get_model
from src.optimizer import OptimizerEnum, get_optimizer
from src.train import train


def main(
        device_str: str,
        fake_path: str,
        true_path: str,
        cache_folder: str,
        batch_size: int,
        n_epoches: int,
        is_half: bool,
        last_states: int,
        arch: ModelEnum,
        optim: OptimizerEnum) -> None:

    device = torch.device(device_str)
    dataset = FakeNewsDataset(fake_path, true_path,
                              device, batch_size, test_size=0.05)

    model = get_model(arch, device, last_states)

    if is_half:
        model = model.half()

    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer, scheduler = get_optimizer(optim, model)

    cache_path = Path(cache_folder)
    cache_path.mkdir(exist_ok=True, parents=True)

    train(
        n_epoches=n_epoches,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        cache_path=cache_path,
        scheduler=scheduler
    )


if __name__ == '__main__':
    args = parse()
    main(args.device, args.fake_path, args.true_path, args.cache_folder,
         args.batch_size, args.epoches, args.is_half, args.last_states, args.arch, args.optim)
