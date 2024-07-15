from typing import Optional
from kb_ops.forward import ForwardOutput, DictDataClass
from dataclasses import dataclass, field

@dataclass
class ValidationOutput(DictDataClass):
    loss: float
    component_accuracy: float
    non_component_accuracy: float

@dataclass
class TrainOutput(DictDataClass):
    best_ckpt_epoch: int = None
    best_ckpt_path: str = None
    train_outputs: list[ForwardOutput] = None
    val_outputs: Optional[list[ValidationOutput]] = field(
        default=None,
        metadata={'description': 'List of outputs from validation dataloader if val_dl is provided'}
    )