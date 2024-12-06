# src/lp_det/config_loader/config.py
"""
config.py
Author: Elias Dargham (edargham)
Team members: Elias Dargham (edargham), Anthony Tabet (Anthony-Tabet) [#DreamTeam]
Date: 2024-12-01
Description: Configuration file for the License Plate Detection model.
"""

import os
from typing import Literal

from pydantic import BaseModel, field_validator
from omegaconf import OmegaConf
import torch

class DataConfig(BaseModel):
    """
    Configuration model for the data pipeline
    of the License Plate Detection model.
    """
    path: str
    bsz: int
    imgsz: int

    @field_validator('path')
    @classmethod
    def check_path(cls, v: str)-> str:
        """
        ### Description
            Check if the path exists.
        ### Parameters
            v (str): The path to check.
        ### Returns
            v (str): The path if it exists.
        ### Raises
            ValueError: If the path does not exist.
        """
        if not isinstance(v, str):
            raise ValueError(f'Value "{v}" is not a string.')
        if not os.path.exists(v):
            raise ValueError(f'Path "{v}" does not exist.')
        return v

    @field_validator('bsz')
    @classmethod
    def check_batch_size(cls, v: int)-> int:
        """
        ### Description
            Check if the batch size is valid.
        ### Parameters
            v (int): The batch size to check.
        ### Returns
            v (int): The batch size if it is valid.
        ### Raises
            ValueError: If the batch size is not valid.
        """
        if not isinstance(v, int):
            raise ValueError(f'Value "{v}" is not an integer.')
        if v <= 0:
            raise ValueError(f'Batch size "{v}" is not valid.')
        return v

    @field_validator('imgsz')
    @classmethod
    def check_img_size(cls, v: int)-> int:
        """
        ### Description
            Check if the image size is valid.
        ### Parameters
            v (int): The image size to check.
        ### Returns
            v (int): The image size if it is valid.
        ### Raises
            ValueError: If the image size is not valid.
        """
        if not isinstance(v, int):
            raise ValueError(f'Value "{v}" is not an integer.')
        if v <= 0:
            raise ValueError(f'Image size "{v}" is not valid.')
        return v

class LearningRateConfig(BaseModel):
    """
    Configuration model for the learning rate scheduler
    of the License Plate Detection model.
    """
    rate: float
    optimizer: Literal['adam', 'sgd', 'auto']
    gamma: float

    @field_validator('rate')
    @classmethod
    def check_rate(cls, v: float)-> float:
        """
        ### Description
            Check if the learning rate is valid.
        ### Parameters
            v (float): The learning rate to check.
        ### Returns
            v (float): The learning rate if it is valid.
        ### Raises
            ValueError: If the learning rate is not valid.
        """
        if not isinstance(v, float):
            raise ValueError(f'Value "{v}" is not a float.')
        if v <= 0:
            raise ValueError(f'Learning rate "{v}" is not valid.')
        return v

    @field_validator('optimizer')
    @classmethod
    def check_optimizer(cls, v: str)-> str:
        """
        ### Description
            Check if the optimizer is valid.
        ### Parameters
            v (str): The optimizer to check.
        ### Returns
            v (str): The optimizer if it is valid.
        ### Raises
            ValueError: If the optimizer is not valid.
        """
        if not isinstance(v, str):
            raise ValueError(f'Value "{v}" is not a string.')
        if v not in ['adam', 'sgd', 'auto']:
            raise ValueError(f'Optimizer "{v}" is not valid.')
        return v

    @field_validator('gamma')
    @classmethod
    def check_gamma(cls, v: float)-> float:
        """
        ### Description
            Check if the gamma is valid.
        ### Parameters
            v (float): The gamma to check.
        ### Returns
            v (float): The gamma if it is valid.
        ### Raises
            ValueError: If the gamma is not valid.
        """
        if not isinstance(v, float):
            raise ValueError(f'Value "{v}" is not a float.')
        if v <= 0:
            raise ValueError(f'Gamma "{v}" is not valid.')
        return v

class TrainingConfig(BaseModel):
    """
    Configuration model for the training pipeline
    of the License Plate Detection model.
    """
    model: Literal['yolov8', 'yolov10', 'yolov11']
    size: Literal['n', 's', 'm', 'l', 'x']
    data: DataConfig
    device: Literal['cpu', 'cuda']
    epochs: int
    workers: int
    lr: LearningRateConfig

    @field_validator('model')
    @classmethod
    def check_model(cls, v: str)-> str:
        """
        ### Description
            Check if the model is valid.
        ### Parameters
            v (str): The model to check.
        ### Returns
            v (str): The model if it is valid.
        ### Raises
            ValueError: If the model is not valid.
        """
        if not isinstance(v, str):
            raise ValueError(f'Value "{v}" is not a string.')
        if v not in ['yolov8', 'yolov10', 'yolov11']:
            raise ValueError(f'Model "{v}" is not valid.')
        return v

    @field_validator('size')
    @classmethod
    def check_size(cls, v: str)-> str:
        """
        ### Description
            Check if the size is valid.
        ### Parameters
            v (str): The size to check.
        ### Returns
            v (str): The size if it is valid.
        ### Raises
            ValueError: If the size is not valid.
        """
        if not isinstance(v, str):
            raise ValueError(f'Value "{v}" is not a string.')
        if v not in ['n', 's', 'm', 'l', 'x']:
            raise ValueError(f'Size "{v}" is not valid.')
        return v

    @field_validator('device')
    @classmethod
    def check_device(cls, v: str)-> str:
        """
        ### Description
            Check if the device is valid.
        ### Parameters
            v (str): The device to check.
        ### Returns
            v (str): The device if it is valid.
        ### Raises
            ValueError: If the device is not valid.
        """
        if not isinstance(v, str):
            raise ValueError(f'Value "{v}" is not a string.')
        if v not in ['cpu', 'cuda']:
            raise ValueError(f'Device "{v}" is not valid.')
        return v if v == 'cuda' and torch.cuda.is_available() else 'cpu'

    @field_validator('epochs')
    @classmethod
    def check_epochs(cls, v: int)-> int:
        """
        ### Description
            Check if the number of epochs is valid.
        ### Parameters
            v (int): The number of epochs to check.
        ### Returns
            v (int): The number of epochs if it is valid.
        ### Raises
            ValueError: If the number of epochs is not valid.
        """
        if not isinstance(v, int):
            raise ValueError(f'Value "{v}" is not an integer.')
        if v <= 0:
            raise ValueError(f'Number of epochs "{v}" is not valid.')
        return v

    @field_validator('workers')
    @classmethod
    def check_workers(cls, v: int)-> int:
        """
        ### Description
            Check if the number of workers is valid.
        ### Parameters
            v (int): The number of workers to check.
        ### Returns
            v (int): The number of workers if it is valid.
        ### Raises
            ValueError: If the number of workers is not valid.
        """
        if not isinstance(v, int):
            raise ValueError(f'Value "{v}" is not an integer.')
        if v <= 0:
            raise ValueError(f'Number of workers "{v}" is not valid.')
        return v

class ValidationConfig(BaseModel):
    """
    Configuration model for the validation pipeline
    of the License Plate Detection model.
    """
    data: DataConfig
    device: Literal['cpu', 'cuda']
    workers: int

    @field_validator('device')
    @classmethod
    def check_device(cls, v: str)-> str:
        """
        ### Description
            Check if the device is valid.
        ### Parameters
            v (str): The device to check.
        ### Returns
            v (str): The device if it is valid.
        ### Raises
            ValueError: If the device is not valid.
        """
        if not isinstance(v, str):
            raise ValueError(f'Value "{v}" is not a string.')
        if v not in ['cpu', 'cuda']:
            raise ValueError(f'Device "{v}" is not valid.')
        return v if v == 'cuda' and torch.cuda.is_available() else 'cpu'

    @field_validator('workers')
    @classmethod
    def check_workers(cls, v: int)-> int:
        """
        ### Description
            Check if the number of workers is valid.
        ### Parameters
            v (int): The number of workers to check.
        ### Returns
            v (int): The number of workers if it is valid.
        ### Raises
            ValueError: If the number of workers is not valid.
        """
        if not isinstance(v, int):
            raise ValueError(f'Value "{v}" is not an integer.')
        if v <= 0:
            raise ValueError(f'Number of workers "{v}" is not valid.')
        return v

class ProjectConfiguration(BaseModel):
    """
    Configuration model for the project of the License Plate Detection model.
    """
    name: str

    @field_validator('name')
    @classmethod
    def check_name(cls, v: str)-> str:
        """
        ### Description
            Check if the project name is valid.
        ### Parameters
            v (str): The project name to check.
        ### Returns
            v (str): The project name if it is valid.
        ### Raises
            ValueError: If the project name is not valid.
        """
        if not isinstance(v, str):
            raise ValueError(f'Value "{v}" is not a string.')
        return v

class Config(BaseModel):
    """
    Configuration model for the License Plate Detection model.
    """
    training: TrainingConfig
    validation: ValidationConfig
    project: ProjectConfiguration
    export_format: Literal['onnx', 'torchscript']

    @field_validator('export_format')
    @classmethod
    def check_export_format(cls, v: str)-> str:
        """
        ### Description
            Check if the export format is valid.
        ### Parameters
            v (str): The export format to check.
        ### Returns
            v (str): The export format if it is valid.
        ### Raises
            ValueError: If the export format is not valid.
        """
        if not isinstance(v, str):
            raise ValueError(f'Value "{v}" is not a string.')
        if v not in ['onnx', 'torchscript']:
            raise ValueError(f'Export format "{v}" is not valid.')
        return v

    @staticmethod
    def from_yaml(path: str)-> 'Config':
        """
        ### Description
            Load the configuration from a YAML file.
        ### Parameters
            path (str): The path to the YAML file.
        ### Returns
            config (Config): The configuration model.
        """
        config = OmegaConf.load(path)
        return Config(**config)

    def to_yaml(self, path: str)-> None:
        """
        ### Description
            Save the configuration to a YAML file.
        ### Parameters
            path (str): The path to the YAML file.
        ### Returns
            None
        """
        OmegaConf.save(self.dict(), path)
