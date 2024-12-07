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
from loguru import logger

logger.add("license_plate_detection.log", rotation="10 MB")   # Each log file is limited to 10 MB

class DataConfig(BaseModel):
    """
    ### Description
        Configuration model for the data pipeline
        of the License Plate Detection model.
    ### Attributes
        path (str): The path to the data.
        bsz (int): The batch size.
        imgsz (int): The image size.
    """
    path: str
    bsz: int
    imgsz: int

    @field_validator('path')
    @classmethod
    def check_path(cls, v: str) -> str:
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
        logger.debug("Validating path...")
        if not isinstance(v, str):
            logger.error(f'Path validation failed: Value "{v}" is not a string.')
            raise ValueError(f'Value "{v}" is not a string.')
        if not os.path.exists(v):
            logger.error(f'Path validation failed: Path "{v}" does not exist.')
            raise ValueError(f'Path "{v}" does not exist.')
        logger.info(f"Path validated successfully: {v}")
        return v

    @field_validator('bsz')
    @classmethod
    def check_batch_size(cls, v: int) -> int:
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
        logger.debug("Validating batch size...")
        if not isinstance(v, int):
            logger.error(f'Batch size validation failed: Value "{v}" is not an integer.')
            raise ValueError(f'Value "{v}" is not an integer.')
        if v <= 0:
            logger.error(f'Batch size validation failed: Batch size "{v}" is not valid.')
            raise ValueError(f'Batch size "{v}" is not valid.')
        logger.info(f"Batch size validated successfully: {v}")
        return v

    @field_validator('imgsz')
    @classmethod
    def check_img_size(cls, v: int) -> int:
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
        logger.debug("Validating image size...")
        if not isinstance(v, int):
            logger.error(f'Image size validation failed: Value "{v}" is not an integer.')
            raise ValueError(f'Value "{v}" is not an integer.')
        if v <= 0:
            logger.error(f'Image size validation failed: Image size "{v}" is not valid.')
            raise ValueError(f'Image size "{v}" is not valid.')
        logger.info(f"Image size validated successfully: {v}")
        return v

class LearningRateConfig(BaseModel):
    """
    ### Description
        Configuration model for the learning rate scheduler
        of the License Plate Detection model.
    ### Attributes
        rate (float): The learning rate.
        optimizer (str): The optimizer to use.
        gamma (float): The gamma value for the learning rate scheduler.
    """
    rate: float
    optimizer: Literal['adam', 'sgd', 'auto']
    gamma: float

    @field_validator('rate')
    @classmethod
    def check_rate(cls, v: float) -> float:
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
        logger.debug("Validating learning rate...")
        if not isinstance(v, float):
            logger.error(f'Learning rate validation failed: Value "{v}" is not a float.')
            raise ValueError(f'Value "{v}" is not a float.')
        if v <= 0:
            logger.error(f'Learning rate validation failed: Learning rate "{v}" is not valid.')
            raise ValueError(f'Learning rate "{v}" is not valid.')
        logger.info(f"Learning rate validated successfully: {v}")
        return v

    @field_validator('optimizer')
    @classmethod
    def check_optimizer(cls, v: str) -> str:
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
        logger.debug("Validating optimizer...")
        if not isinstance(v, str):
            logger.error(f'Optimizer validation failed: Value "{v}" is not a string.')
            raise ValueError(f'Value "{v}" is not a string.')
        if v not in ['adam', 'sgd', 'auto']:
            logger.error(f'Optimizer validation failed: Optimizer "{v}" is not valid.')
            raise ValueError(f'Optimizer "{v}" is not valid.')
        logger.info(f"Optimizer validated successfully: {v}")
        return v

    @field_validator('gamma')
    @classmethod
    def check_gamma(cls, v: float) -> float:
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
        logger.debug("Validating gamma...")
        if not isinstance(v, float):
            logger.error(f'Gamma validation failed: Value "{v}" is not a float.')
            raise ValueError(f'Value "{v}" is not a float.')
        if v <= 0:
            logger.error(f'Gamma validation failed: Gamma "{v}" is not valid.')
            raise ValueError(f'Gamma "{v}" is not valid.')
        logger.info(f"Gamma validated successfully: {v}")
        return v

class TrainingConfig(BaseModel):
    """
    ### Description
        Configuration model for the training pipeline
        of the License Plate Detection model.
    ### Attributes
        model (str): The model to use for training.
        size (str): The size of the model to use for training.
        data (DataConfig): The data pipeline configuration.
        device (str): The device to use for training.
        epochs (int): The number of epochs to train for.
        workers (int): The number of CPU workers to use for training.
        lr (LearningRateConfig): The learning rate scheduler configuration.
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
    def check_model(cls, v: str) -> str:
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
        logger.debug("Validating model...")
        if not isinstance(v, str):
            logger.error(f'Model validation failed: Value "{v}" is not a string.')
            raise ValueError(f'Value "{v}" is not a string.')
        if v not in ['yolov8', 'yolov10', 'yolov11']:
            logger.error(f'Model validation failed: Model "{v}" is not valid.')
            raise ValueError(f'Model "{v}" is not valid.')
        logger.info(f"Model validated successfully: {v}")
        return v

    @field_validator('size')
    @classmethod
    def check_size(cls, v: str) -> str:
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
        logger.debug("Validating size...")
        if not isinstance(v, str):
            logger.error(f'Size validation failed: Value "{v}" is not a string.')
            raise ValueError(f'Value "{v}" is not a string.')
        if v not in ['n', 's', 'm', 'l', 'x']:
            logger.error(f'Size validation failed: Size "{v}" is not valid.')
            raise ValueError(f'Size "{v}" is not valid.')
        logger.info(f"Size validated successfully: {v}")
        return v

    @field_validator('device')
    @classmethod
    def check_device(cls, v: str) -> str:
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
        logger.debug("Validating device...")
        if not isinstance(v, str):
            logger.error(f'Device validation failed: Value "{v}" is not a string.')
            raise ValueError(f'Value "{v}" is not a string.')
        if v not in ['cpu', 'cuda']:
            logger.error(f'Device validation failed: Device "{v}" is not valid.')
            raise ValueError(f'Device "{v}" is not valid.')
        validated_device = v if v == 'cuda' and torch.cuda.is_available() else 'cpu'
        logger.info(f"Device validated successfully: {validated_device}")
        return validated_device

    @field_validator('epochs')
    @classmethod
    def check_epochs(cls, v: int) -> int:
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
        logger.debug("Validating number of epochs...")
        if not isinstance(v, int):
            logger.error(f'Epochs validation failed: Value "{v}" is not an integer.')
            raise ValueError(f'Value "{v}" is not an integer.')
        if v <= 0:
            logger.error(f'Epochs validation failed: Number of epochs "{v}" is not valid.')
            raise ValueError(f'Number of epochs "{v}" is not valid.')
        logger.info(f"Number of epochs validated successfully: {v}")
        return v

    @field_validator('workers')
    @classmethod
    def check_workers(cls, v: int) -> int:
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
        logger.debug("Validating number of workers...")
        if not isinstance(v, int):
            logger.error(f'Workers validation failed: Value "{v}" is not an integer.')
            raise ValueError(f'Value "{v}" is not an integer.')
        if v <= 0:
            logger.error(f'Workers validation failed: Number of workers "{v}" is not valid.')
            raise ValueError(f'Number of workers "{v}" is not valid.')
        logger.info(f"Number of workers validated successfully: {v}")
        return v

class ValidationConfig(BaseModel):
    """
    ### Description
        Configuration model for the validation pipeline
        of the License Plate Detection model.
    ### Attributes
        data (DataConfig): The data pipeline configuration.
        device (str): The device to use for validation.
        workers (int): The number of CPU workers to use for validation.
    """
    data: DataConfig
    device: Literal['cpu', 'cuda']
    workers: int

    @field_validator('device')
    @classmethod
    def check_device(cls, v: str) -> str:
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
        logger.debug("Validating device for validation pipeline...")
        if not isinstance(v, str):
            logger.error(f'Device validation failed: Value "{v}" is not a string.')
            raise ValueError(f'Value "{v}" is not a string.')
        if v not in ['cpu', 'cuda']:
            logger.error(f'Device validation failed: Device "{v}" is not valid.')
            raise ValueError(f'Device "{v}" is not valid.')
        validated_device = v if v == 'cuda' and torch.cuda.is_available() else 'cpu'
        logger.info(f"Device validated successfully for validation pipeline: {validated_device}")
        return validated_device

    @field_validator('workers')
    @classmethod
    def check_workers(cls, v: int) -> int:
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
        logger.debug("Validating number of workers for validation pipeline...")
        if not isinstance(v, int):
            logger.error(f'Workers validation failed: Value "{v}" is not an integer.')
            raise ValueError(f'Value "{v}" is not an integer.')
        if v <= 0:
            logger.error(f'Workers validation failed: Number of workers "{v}" is not valid.')
            raise ValueError(f'Number of workers "{v}" is not valid.')
        logger.info(f"Number of workers validated successfully for validation pipeline: {v}")
        return v

class ProjectConfiguration(BaseModel):
    """
    ### Description
        Configuration model for the project of the License Plate Detection model.
    ### Attributes
        name (str): The name of the project.
    """
    name: str

    @field_validator('name')
    @classmethod
    def check_name(cls, v: str) -> str:
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
        logger.debug("Validating project name...")
        if not isinstance(v, str):
            logger.error(f'Project name validation failed: Value "{v}" is not a string.')
            raise ValueError(f'Value "{v}" is not a string.')
        logger.info(f"Project name validated successfully: {v}")
        return v

class Config(BaseModel):
    """
    ### Description
        Configuration model for the License Plate Detection model.
    ### Attributes
        training (TrainingConfig): The training pipeline configuration.
        validation (ValidationConfig): The validation pipeline configuration.
        project (ProjectConfiguration): The project configuration.
        export_format (str): The export format for the model.
    """
    training: TrainingConfig
    validation: ValidationConfig
    project: ProjectConfiguration
    export_format: Literal['onnx', 'torchscript']

    @field_validator('export_format')
    @classmethod
    def check_export_format(cls, v: str) -> str:
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
        logger.debug("Validating export format...")
        if not isinstance(v, str):
            logger.error(f'Export format validation failed: Value "{v}" is not a string.')
            raise ValueError(f'Value "{v}" is not a string.')
        if v not in ['onnx', 'torchscript']:
            logger.error(f'Export format validation failed: Export format "{v}" is not valid.')
            raise ValueError(f'Export format "{v}" is not valid.')
        logger.info(f"Export format validated successfully: {v}")
        return v

    @staticmethod
    def from_yaml(path: str) -> 'Config':
        """
        ### Description
            Load the configuration from a YAML file.
        ### Parameters
            path (str): The path to the YAML file.
        ### Returns
            config (Config): The configuration model.
        """
        logger.debug(f"Loading configuration from YAML file: {path}")
        try:
            config = OmegaConf.load(path)
            logger.info(f"Configuration loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {path}: {e}")
            raise
        return Config(**config)

    def to_yaml(self, path: str) -> None:
        """
        ### Description
            Save the configuration to a YAML file.
        ### Parameters
            path (str): The path to the YAML file.
        ### Returns
            None
        """
        logger.debug(f"Saving configuration to YAML file: {path}")
        try:
            OmegaConf.save(self.model_dump(), path)
            logger.info(f"Configuration saved successfully to {path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {path}: {e}")
            raise

    def to_dict(self) -> dict:
        """
        ### Description
            Convert the configuration to a dictionary.
        ### Returns
            config (dict): The configuration dictionary.
        """
        config = self.model_dump()
        if not OmegaConf.is_config(config):
            config = OmegaConf.create(config)
        return OmegaConf.to_container(config)
