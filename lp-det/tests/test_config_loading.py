# tests/test_config_loading.py
"""
test_config_loading.py
Author: Elias Dargham (edargham)
Date: 2024-12-15
Description: Tests for the configuration loading module.
"""

import pytest
from pydantic import ValidationError
from src.detector_training.config_loader.config\
    import DataConfig, LearningRateConfig, TrainingConfig, ValidationConfig, ProjectConfiguration, Config


def test_valid_data_config():
    config = DataConfig(path="config/config-dev.yaml", bsz=32, imgsz=640)
    assert config.path == "config/config-dev.yaml"
    assert config.bsz == 32
    assert config.imgsz == 640

def test_invalid_path_non_string():
    with pytest.raises(ValidationError):
        DataConfig(path=123, bsz=32, imgsz=640)

def test_invalid_path_non_existent():
    with pytest.raises(ValidationError):
        DataConfig(path="/non/existent/path", bsz=32, imgsz=640)

def test_invalid_batch_size_non_integer():
    with pytest.raises(ValidationError):
        DataConfig(path="config/config-dev.yaml", bsz="thirty-two", imgsz=640)

def test_invalid_batch_size_non_positive():
    with pytest.raises(ValidationError):
        DataConfig(path="config/config-dev.yaml", bsz=0, imgsz=640)

def test_invalid_image_size_non_integer():
    with pytest.raises(ValidationError):
        DataConfig(path="config/config-dev.yaml", bsz=32, imgsz="six hundred and forty")

def test_invalid_image_size_non_positive():
    with pytest.raises(ValidationError):
        DataConfig(path="config/config-dev.yaml", bsz=32, imgsz=0)

def test_valid_learning_rate_config():
    config = LearningRateConfig(rate=0.001, optimizer='adam', gamma=0.9)
    assert config.rate == 0.001
    assert config.optimizer == 'adam'
    assert config.gamma == 0.9

def test_invalid_rate_non_float():
    with pytest.raises(ValidationError):
        LearningRateConfig(rate="one thousandth", optimizer='adam', gamma=0.9)

def test_invalid_rate_non_positive():
    with pytest.raises(ValidationError):
        LearningRateConfig(rate=-0.001, optimizer='adam', gamma=0.9)

def test_invalid_optimizer_non_string():
    with pytest.raises(ValidationError):
        LearningRateConfig(rate=0.001, optimizer=123, gamma=0.9)

def test_invalid_optimizer_value():
    with pytest.raises(ValidationError):
        LearningRateConfig(rate=0.001, optimizer='invalid_optimizer', gamma=0.9)

def test_invalid_gamma_non_float():
    with pytest.raises(ValidationError):
        LearningRateConfig(rate=0.001, optimizer='adam', gamma="nine tenths")

def test_invalid_gamma_non_positive():
    with pytest.raises(ValidationError):
        LearningRateConfig(rate=0.001, optimizer='adam', gamma=-0.9)

def test_training_config_valid():
    data_config = DataConfig(path="config/config-dev.yaml", bsz=32, imgsz=640)
    lr_config = LearningRateConfig(rate=0.001, optimizer="adam", gamma=0.9)
    config = TrainingConfig(
        model="yolov8",
        size="m",
        data=data_config,
        device="cuda",
        epochs=10,
        workers=4,
        lr=lr_config
    )
    assert config.model == "yolov8"
    assert config.size == "m"
    assert config.data == data_config
    assert config.device == "cuda"
    assert config.epochs == 10
    assert config.workers == 4
    assert config.lr == lr_config

def test_training_config_invalid_model():
    data_config = DataConfig(path="config/config-dev.yaml", bsz=32, imgsz=640)
    lr_config = LearningRateConfig(rate=0.001, optimizer="adam", gamma=0.9)
    with pytest.raises(ValidationError):
        TrainingConfig(
            model="invalid_model",
            size="m",
            data=data_config,
            device="cuda",
            epochs=10,
            workers=4,
            lr=lr_config
        )

def test_training_config_invalid_size():
    data_config = DataConfig(path="config/config-dev.yaml", bsz=32, imgsz=640)
    lr_config = LearningRateConfig(rate=0.001, optimizer="adam", gamma=0.9)
    with pytest.raises(ValidationError):
        TrainingConfig(
            model="yolov8",
            size="invalid_size",
            data=data_config,
            device="cuda",
            epochs=10,
            workers=4,
            lr=lr_config
        )

def test_training_config_invalid_device():
    data_config = DataConfig(path="config/config-dev.yaml", bsz=32, imgsz=640)
    lr_config = LearningRateConfig(rate=0.001, optimizer="adam", gamma=0.9)
    with pytest.raises(ValidationError):
        TrainingConfig(
            model="yolov8",
            size="m",
            data=data_config,
            device="invalid_device",
            epochs=10,
            workers=4,
            lr=lr_config
        )

def test_training_config_invalid_epochs():
    data_config = DataConfig(path="config/config-dev.yaml", bsz=32, imgsz=640)
    lr_config = LearningRateConfig(rate=0.001, optimizer="adam", gamma=0.9)
    with pytest.raises(ValidationError):
        TrainingConfig(
            model="yolov8",
            size="m",
            data=data_config,
            device="cuda",
            epochs=-1,
            workers=4,
            lr=lr_config
        )

def test_training_config_invalid_workers():
    data_config = DataConfig(path="config/config-dev.yaml", bsz=32, imgsz=640)
    lr_config = LearningRateConfig(rate=0.001, optimizer="adam", gamma=0.9)
    with pytest.raises(ValidationError):
        TrainingConfig(
            model="yolov8",
            size="m",
            data=data_config,
            device="cuda",
            epochs=10,
            workers=-1,
            lr=lr_config
        )

def test_validation_config_valid():
    data_config = DataConfig(path="config/config-dev.yaml", bsz=32, imgsz=640)
    config = ValidationConfig(
        data=data_config,
        device="cuda",
        workers=4
    )
    assert config.data == data_config
    assert config.device == "cuda"
    assert config.workers == 4

def test_validation_config_invalid_device():
    data_config = DataConfig(path="config/config-dev.yaml", bsz=32, imgsz=640)
    with pytest.raises(ValidationError):
        ValidationConfig(
            data=data_config,
            device="invalid_device",
            workers=4
        )

def test_validation_config_invalid_workers():
    data_config = DataConfig(path="config/config-dev.yaml", bsz=32, imgsz=640)
    with pytest.raises(ValidationError):
        ValidationConfig(
            data=data_config,
            device="cuda",
            workers=-1
        )

def test_project_configuration_valid():
    config = ProjectConfiguration(name="License Plate Detection")
    assert config.name == "License Plate Detection"

def test_project_configuration_invalid_name():
    with pytest.raises(ValidationError):
        ProjectConfiguration(name=123)

def test_config_valid():
    data_config = DataConfig(path="config/config-dev.yaml", bsz=32, imgsz=640)
    lr_config = LearningRateConfig(rate=0.001, optimizer="adam", gamma=0.9)
    training_config = TrainingConfig(
        model="yolov8",
        size="m",
        data=data_config,
        device="cuda",
        epochs=10,
        workers=4,
        lr=lr_config
    )
    validation_config = ValidationConfig(
        data=data_config,
        device="cuda",
        workers=4
    )
    project_config = ProjectConfiguration(name="License Plate Detection")
    config = Config(
        training=training_config,
        validation=validation_config,
        project=project_config,
        export_format="onnx"
    )
    assert config.training == training_config
    assert config.validation == validation_config
    assert config.project == project_config
    assert config.export_format == "onnx"

def test_config_invalid_export_format():
    data_config = DataConfig(path="config/config-dev.yaml", bsz=32, imgsz=640)
    lr_config = LearningRateConfig(rate=0.001, optimizer="adam", gamma=0.9)
    training_config = TrainingConfig(
        model="yolov8",
        size="m",
        data=data_config,
        device="cuda",
        epochs=10,
        workers=4,
        lr=lr_config
    )
    validation_config = ValidationConfig(
        data=data_config,
        device="cuda",
        workers=4
    )
    project_config = ProjectConfiguration(name="License Plate Detection")
    with pytest.raises(ValidationError):
        Config(
            training=training_config,
            validation=validation_config,
            project=project_config,
            export_format="invalid_format"
        )
