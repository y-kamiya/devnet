from dataclasses import dataclass
from enum import Enum, auto

import torch


class LogLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARN = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class TrainerConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    loglevel: LogLevel = LogLevel.INFO
    dataroot: str = "data"
    batch_size: int = 512
    n_batch: int = 20
    epochs: int = 1
    tensorboard_log_dir: str = "${dataroot}/runs/${name}"
    model_path: str = "${dataroot}/${name}.pth"
    log_interval: int = 1
    eval_interval: int = 1
    random_seed: int = 42
