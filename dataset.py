from typing import Generator, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


class FakeNewsDataset:
    
    def __init__(self, fake_path: str, true_path: str, device: torch.device, batch_size: int) -> None:
        self._batch_size = batch_size
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        fake_raw = pd.read_csv(fake_path)
        true_raw = pd.read_csv(true_path)

        fake_raw['is_fake'] = 1
        true_raw['is_fake'] = 0
        concat = pd.concat([fake_raw, true_raw])

        # shuffle data
        dataset = concat.sample(frac=1).reset_index(drop=True)

        data_array = list(dataset['text'])
        target_array = list(dataset['is_fake'])

        self._X_train, self._X_val, self._y_train, self._y_val = train_test_split(data_array, target_array, test_size = 0.20, random_state = 42)

    def _process(self, data_batch, target_batch) -> Tuple[dict, torch.Tensor]:
        data = self._tokenizer(data_batch, padding = True, truncation = True, return_tensors="pt")
        data = {k:torch.tensor(v).to(self._device) for k,v in data.items()}
        targets = torch.tensor(target_batch, dtype=torch.float32).view(-1, 1).to(self._device) # flatten
        return data, targets

    def get_batches(self, is_train: bool = True):
        if is_train:
            data = self._X_train
            targets = self._y_train
        else:
            data = self._X_val
            targets = self._y_val

        data_batches = list(self._generate_batch(data))
        target_batches = list(self._generate_batch(targets))

        processed_data_batches = []
        processed_target_batches = []
        for data_batch, target_batch in zip(data_batches, target_batches):
            data, target = self._process(data_batch, target_batch)
            processed_data_batches.append(data)
            processed_target_batches.append(target)

        return list(zip(processed_data_batches, processed_target_batches))

    def _generate_batch(self, lst: list) -> Generator:
        """  Yields batch of specified size """
        for i in range(0, len(lst), self._batch_size):
            yield lst[i : i + self._batch_size]
