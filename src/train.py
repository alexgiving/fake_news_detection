import time
from pathlib import Path

import torch
from typing import Optional

from src.dataset import FakeNewsDataset
from src.model import BaseModel


def batch_logging(mode: str, batch_id: int, total_batch: int, output_time: float) -> None:
    print(f'Processed {mode} [{batch_id}/{total_batch}] batch time: {output_time:.4f}', flush=True)


def train(
        n_epoches: int, 
        dataset: FakeNewsDataset,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        cache_path: Path,
        scheduler: Optional[torch.optim.lr_scheduler.StepLR] = None,
    ) -> None:

    print(f'Model: {type(model)}, \n Optimizer: {optimizer}')
        

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

        model.eval()
        total = 0
        corrects = 0
        for batch_id, (data, target) in enumerate(val_data):
            batch_time = time.perf_counter()
            predict = model.infer(data)
            corrects += torch.sum(predict == target).cpu().item()
            total += predict.shape[1]
            batch_logging('val', batch_id, n_val_batches, time.perf_counter() - batch_time)

        optimizer.step()
        if scheduler: scheduler.step() 

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
