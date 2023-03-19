from enum import Enum
from typing import Optional, Tuple

import torch

from src.model import BaseModel


class OptimizerEnum(Enum):
    SGD = 'sgd'
    ADAM = 'adam'

    def __str__(self):
        return self.value


def get_optimizer(optim: OptimizerEnum, model: BaseModel) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.StepLR]]:
    if optim is OptimizerEnum.SGD:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, weight_decay=0.001, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1)
        return optimizer, scheduler
    elif optim is OptimizerEnum.ADAM:
        optimizer = torch.optim.Adam(params=model.parameters())
        return optimizer, None
    else:
        raise KeyError('Not supported optimizer')
