from pathlib import Path

import torch

from src.arguments import parse
from src.dataset import FakeNewsDataset
from src.model import BertBasedClassificationModel
from src.train import train


def main(device_str: str, fake_path: str, true_path: str, cache_folder: str, batch_size: int, n_epoches: int, is_half: bool) -> None:
    device = torch.device(device_str)
    dataset = FakeNewsDataset(fake_path, true_path, device, batch_size, test_size=0.05)

    model = BertBasedClassificationModel(device)
    if is_half:
        model = model.half()

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    cache_path = Path(cache_folder)
    cache_path.mkdir(exist_ok=True, parents=True)

    train(n_epoches, dataset, model, optimizer, criterion, scheduler, cache_path)


if __name__ == '__main__':
    args = parse()
    main(args.device, args.fake_path, args.true_path, args.cache_folder, args.batch_size, args.epoches, args.is_half)
