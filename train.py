import time
from pathlib import Path

import torch

from config import (batch_size, cache_folder, device_name, fake_path,
                    is_half_precision, milestones, n_epoches, test_size,
                    true_path)
from dataset import FakeNewsDataset
from model import BertBasedClassificationModel


def batch_logging(mode: str, batch_id: int, total_batch: int, output_time: float) -> None:
    print(f'Processed {mode} [{batch_id}/{total_batch}] batch time: {output_time:.4f}', flush=True)


def train(
        n_epoches: int, 
        dataset: FakeNewsDataset,
        model: BertBasedClassificationModel,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler.MultiStepLR,
        cache_path: Path
    ) -> None:

    train_data = dataset.get_batches(is_train=True)
    val_data = dataset.get_batches(is_train=False)

    n_train_batches = len(train_data)
    n_val_batches = len(val_data)

    best_accuracy = 0
    training_time = time.perf_counter()
    for epoch_id in range(1, n_epoches+1):
        optimizer.zero_grad()
        epoch_time = time.perf_counter()

        model.train()
        for batch_id, (data, target) in enumerate(train_data):
            batch_time = time.perf_counter()
            predict = model.forward(data)
            loss = criterion(predict, target)
            loss.backward()
            batch_logging('train', batch_id, n_train_batches, time.perf_counter() - batch_time)
            if batch_id == 2: break

        model.eval()
        total = 0
        corrects = 0
        for batch_id, (data, target) in enumerate(val_data):
            batch_time = time.perf_counter()
            predict = model.infer(data)
            corrects += torch.sum(predict == target).cpu().item()
            total += predict.shape[1]
            batch_logging('val', batch_id, n_val_batches, time.perf_counter() - batch_time)
            if batch_id == 2: break

        optimizer.step()
        scheduler.step()

        accuracy = corrects/total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            ckpt_path = cache_path / f'model_acc_{accuracy:.4f}_epoch_{epoch_id}.ckpt'
            torch.save(model.state_dict(), ckpt_path)

        end_epoch_time = time.perf_counter()
        print(
            f'Epoch [{epoch_id}/{n_epoches+1}] ',
            f'Epoch time: {end_epoch_time - epoch_time:.4f} ',
            f'Total time: {end_epoch_time - training_time:.4f}',
            f'Accuracy: {accuracy:.4f}',
            f'Best accuracy {best_accuracy:.4f}', flush=True)

    print(f'Total time: {time.perf_counter() - training_time}', flush=True)
    print(f'Best val Acc: {best_accuracy:.4f}')


def main() -> None:
    device = torch.device(device_name)
    dataset = FakeNewsDataset(fake_path, true_path, device, batch_size, test_size)

    model = BertBasedClassificationModel(device)
    if is_half_precision:
        model = model.half()

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    cache_path = Path(cache_folder)
    cache_path.mkdir(exist_ok=True, parents=True)

    train(n_epoches, dataset, model, optimizer, criterion, scheduler, cache_path)


if __name__ == '__main__':
    main()
