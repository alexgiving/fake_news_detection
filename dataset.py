import pandas as pd
import torch
from transformers import AutoTokenizer
import numpy as np
from sklearn.model_selection import train_test_split



class FakeNewsDataset:
    
    def __init__(self, fake_path: str, true_path: str, device: torch.device) -> None:
        pass