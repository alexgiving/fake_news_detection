import time

import torch

from config import n_epoches, batch_size, milestones, device_name, fake_path, true_path, test_size
from dataset import FakeNewsDataset
from model import BertBasedClassificationModel


def batch_logging(mode: str, batch_id: int, total_batch: int, output_time: float):
    print(f'Processed {mode} [{batch_id}/{total_batch}] batch time: {output_time:.4f}', flush=True)


def main():
    device = torch.device(device_name)
    dataset = FakeNewsDataset(fake_path, true_path, device, batch_size, test_size)

    model = BertBasedClassificationModel(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    train(n_epoches, dataset, model, optimizer, criterion, scheduler)


def train(
        n_epoches: int, 
        dataset: FakeNewsDataset,
        model: BertBasedClassificationModel,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler.MultiStepLR
    ) -> None:

    train_data = dataset.get_batches(is_train=True)
    val_data = dataset.get_batches(is_train=False)

    n_train_batches = len(train_data)
    n_val_batches = len(val_data)

    best_accuracy = 0
    training_time = time.perf_counter()
    for epoch_id in range(1, n_epoches+1):
        epoch_time = time.perf_counter()

        model.train()
        for batch_id, (data, target) in enumerate(train_data):
            batch_time = time.perf_counter()
            optimizer.zero_grad()
            predict = model.forward(data)
            loss = criterion(predict, target)
            loss.backward()
            batch_logging('train', batch_id, n_train_batches, time.perf_counter() - batch_time)

        model.eval()
        total = 0
        corrects = 0
        for batch_id, (data, target) in enumerate(val_data):
            batch_time = time.perf_counter()
            predict = model.infer(data)
            corrects += torch.sum(predict == target).cpu()
            total += len(predict)
            batch_logging('val', batch_id, n_val_batches, time.perf_counter() - batch_time)

        scheduler.step()

        accuracy = corrects/total
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        end_epoch_time = time.perf_counter()
        print(
            f'Epoch [{epoch_id}/{n_epoches+1}] ',
            f'Epoch time: {end_epoch_time - epoch_time:.4f} ',
            f'Total time: {end_epoch_time - training_time:.4f}',
            f'Accuracy: {accuracy:.4f}',
            f'Best accuracy {best_accuracy}', flush=True)

    print(f'Total time: {time.perf_counter() - training_time}', flush=True)
    print(f'Best val Acc: {best_accuracy:.4f}')


if __name__ == '__main__':
    main()
