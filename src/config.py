from dataclasses import dataclass


@dataclass
class TrainConfig:
    learning_rate: float
    momentum: float
    n_epochs: int
    in_channels: int
    n_classes: int
    batch_size: int
    device: str
    seed: int

    eval_every: int=1
    save_every: int=1
    visualize_every: int=1
    verbose: bool=True