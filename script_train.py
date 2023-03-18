from pathlib import Path

import torch

from src.arguments import parse
from src.dataset import FakeNewsDataset
from src.model import ClassBertModel, NormalizedClassBertModel, DFCNormalizedClassBertModel
from src.train import train


def main(
        device_str: str,
        fake_path: str,
        true_path: str,
        cache_folder: str,
        batch_size: int,
        n_epoches: int,
        is_half: bool,
        last_states: int) -> None:

    device = torch.device(device_str)
    dataset = FakeNewsDataset(fake_path, true_path,
                              device, batch_size, test_size=0.05)

    model = DFCNormalizedClassBertModel(device, last_states=last_states)
    if is_half:
        model = model.half()

    criterion = torch.nn.CrossEntropyLoss().to(device)

    ##########

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    optimizer = torch.optim.Adam(params=model.parameters())
    scheduler = None

    ##########

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
         args.batch_size, args.epoches, args.is_half, args.last_states)
