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
        params = f'Model: {type(self)}, Last states: {self._last_states}, Feature extractor: {type(self._feature_extractor)}'
        return params


class ClassBertModel(BaseModel):

    def __init__(self, device: torch.device, last_states: int = 1) -> None:
        super().__init__(device, last_states)
        self._fc = torch.nn.Linear(768, 1).to(self._device)

    def forward(self, input_batch) -> torch.Tensor:
        output = self.get_embedding(input_batch)
        output = self._fc(output)
        return output.view(1, -1)


class NormalizedClassBertModel(BaseModel):

    def __init__(self, device: torch.device, last_states: int = 1) -> None:
        super().__init__(device, last_states)
        self._fc = torch.nn.Linear(768, 1).to(self._device)
        self._bn = torch.nn.BatchNorm1d(768).to(self._device)

    def forward(self, input_batch) -> torch.Tensor:
        output = self.get_embedding(input_batch)
        output = self._bn(output)
        output = self._fc(output)
        return output.view(1, -1)
    

class DFCNormalizedClassBertModel(BaseModel):

    def __init__(self, device: torch.device, last_states: int = 1) -> None:
        super().__init__(device, last_states)
        self._fc = torch.nn.Linear(768, 1).to(self._device)
        self._fc_hidden = torch.nn.Linear(768, 768).to(self._device)
        self._bn = torch.nn.BatchNorm1d(768).to(self._device)

    def forward(self, input_batch) -> torch.Tensor:
        output = self.get_embedding(input_batch)
        output = self._bn(output)
        output = self._fc_hidden(output)
        output = self._fc(output)
        return output.view(1, -1)
