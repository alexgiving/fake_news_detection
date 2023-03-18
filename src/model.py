import torch
from transformers import AutoModel, logging

logging.set_verbosity_error()

class BaseModel(torch.nn.Module):

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self._device = device
        self._feature_extractor = AutoModel.from_pretrained("distilbert-base-uncased").to(self._device)
        self._fc_hidden = torch.nn.Linear(768, 768).to(self._device)
        self._fc = torch.nn.Linear(768, 1).to(self._device)
        self._activ = torch.nn.Sigmoid().to(self._device)
        self._bn = torch.nn.BatchNorm1d(768).to(self._device)

    @torch.no_grad()
    def get_embedding(self, input_batch) -> torch.Tensor:
        # dim : [batch_size(nr_sentences), tokens, emb_dim]
        data = self._feature_extractor(**input_batch)
        new_data = data.last_hidden_state[:, -1, :] + data.last_hidden_state[:, -2, :] + data.last_hidden_state[:, -3, :] + data.last_hidden_state[:, -4, :]
        return new_data

    def forward(self, input_batch) -> torch.Tensor:
        ...

    @torch.no_grad()
    def infer(self, input_batch) -> torch.Tensor:
        output = self.forward(input_batch)
        return torch.as_tensor((output - 0.5) > 0, dtype=torch.int32).to(self._device)


class ClassBertModel(BaseModel):

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)

    def forward(self, input_batch) -> torch.Tensor:
        output = self.get_embedding(input_batch)
        output = self._fc(output)
        return output.view(1, -1)


class NormalizedClassBertModel(BaseModel):

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)

    def forward(self, input_batch) -> torch.Tensor:
        output = self.get_embedding(input_batch)
        output = self._bn(output)
        output = self._fc(output)
        return output.view(1, -1)
    

class DoubleHeadNormalizedClassBertModel(BaseModel):

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)

    def forward(self, input_batch) -> torch.Tensor:
        output = self.get_embedding(input_batch)
        output = self._bn(output)
        output = self._fc_hidden(output)
        output = self._fc(output)
        return output.view(1, -1)
