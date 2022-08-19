from dataclasses import dataclass


@dataclass
class Files:
    training: list[list[str]]
    validation: list[list[str]]


@dataclass
class Config:
    num_epochs: int
    batch_size: int
    samples_per_volume: int
    val_batch_size: int
    patch_size: int
    learning_rate: float
    internal_voxsize: float
    data_dir: str
    num_workers: int
    num_feat: int
    num_blocks: int
    num_slices: int
    num_input_ch: int
    val_metric_period: int
    files: Files
