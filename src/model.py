from enum import Enum

import torch
from transformers import AutoModel, logging

logging.set_verbosity_error()


class BaseModel(torch.nn.Module):

    def __init__(self, device: torch.device, last_states: int = 1) -> None:
        super().__init__()
        self._device = device
        self._feature_extractor = AutoModel.from_pretrained("distilbert-base-uncased").to(self._device)
        self._last_states = last_states

    @torch.no_grad()
    def get_embedding(self, input_batch) -> torch.Tensor:
        # dim : [batch_size(nr_sentences), tokens, emb_dim]
        data = self._feature_extractor(**input_batch)

        new_data = data.last_hidden_state[:, -1, :]
        for state_id in range(2, self._last_states + 1):
            # Ex: _last_states = 4 -> -1,-2,-3,-4 is used
            new_data += data.last_hidden_state[:, -state_id, :]
        return new_data

    def forward(self, input_batch) -> torch.Tensor:
        ...

    @torch.no_grad()
    def infer(self, input_batch) -> torch.Tensor:
        output = self.forward(input_batch)
        return torch.as_tensor((output - 0.5) > 0, dtype=torch.int32).to(self._device)
    
    def info(self) -> str:
        params = f'Model_{self.name}_Last_states_{self._last_states}'
        return params


class ClassBertModel(BaseModel):

    def __init__(self, device: torch.device, last_states: int = 1) -> None:
        super().__init__(device, last_states)
        self._fc = torch.nn.Linear(768, 1).to(self._device)
    
    @property
    def name(self) -> str:
        return 'ClassBertModel'

    def forward(self, input_batch) -> torch.Tensor:
        output = self.get_embedding(input_batch)
        output = self._fc(output)
        return output.view(1, -1)


class NormalizedClassBertModel(BaseModel):

    def __init__(self, device: torch.device, last_states: int = 1) -> None:
        super().__init__(device, last_states)
        self._fc = torch.nn.Linear(768, 1).to(self._device)
        self._bn = torch.nn.BatchNorm1d(768).to(self._device)

    @property
    def name(self) -> str:
        return 'NormalizedClassBertModel'

    def forward(self, input_batch) -> torch.Tensor:
        output = self.get_embedding(input_batch)
        output = self._bn(output)
        output = self._fc(output)
        return output.view(1, -1)
    

class DeepNormalizedClassBert(BaseModel):

    def __init__(self, device: torch.device, last_states: int = 1) -> None:
        super().__init__(device, last_states)
        self._fc = torch.nn.Linear(768, 1).to(self._device)
        self._fc_hidden = torch.nn.Linear(768, 768).to(self._device)
        self._bn = torch.nn.BatchNorm1d(768).to(self._device)

    @property
    def name(self) -> str:
        return 'DeepNormalizedClassBert'

    def forward(self, input_batch) -> torch.Tensor:
        output = self.get_embedding(input_batch)
        output = self._bn(output)
        output = self._fc_hidden(output)
        output = self._fc(output)
        return output.view(1, -1)


class ModelEnum(Enum):
    ClassBert = 'class_bert'
    NormalizedClassBert = 'normalized_class_bert'
    DeepNormalizedClassBert = 'deep_normalized_class_bert'

    def __str__(self):
        return self.value


def get_model(arch: ModelEnum, device: torch.device, last_states: int) -> BaseModel:
    if arch is ModelEnum.ClassBert:
        return ClassBertModel(device, last_states=last_states)
    elif arch is ModelEnum.NormalizedClassBert:
        return NormalizedClassBertModel(device, last_states=last_states)
    elif arch is ModelEnum.DeepNormalizedClassBert:
        return DeepNormalizedClassBert(device, last_states=last_states)
    else:
        raise KeyError('Not supported model')
