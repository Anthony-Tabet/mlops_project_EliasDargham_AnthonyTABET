# src/lp_det/main.py
"""
main.py
Author: Elias Dargham (edargham)
Team members: Elias Dargham (edargham), Anthony Tabet (Anthony-Tabet) [#DreamTeam]
Date: 2024-12-01
Description: Main file for the training pipeline of the License Plate Detection model.
"""

import argparse
from dotenv import load_dotenv
from loguru import logger
from ultralytics import YOLO
import mlflow

from detector_training.config_loader import Config


# Setup Loguru logger
logger.add("lp_det.log", rotation="10 MB")   # Each log file is limited to 10 MB

def run(conf: Config) -> None:
    """
    ### Description
        Main function for the training pipeline of the License Plate Detection model.
    ### Parameters
        conf (Config): Configuration instance for the training pipeline.
    ### Returns
        None
    """
    logger.info(f"Setting up MLflow experiment: {conf.project.name}")
    mlflow.set_experiment(conf.project.name)

    # Load the model
    model = YOLO(model=f'{conf.training.model}{conf.training.size}.pt')
    model.model_name = 'lp-det'
    logger.info("Model loaded and configured.")

    # Register the model with mlflow
    mlflow.pytorch.log_model(model, 'model')
    logger.info("Model registered in MLflow.")

    # Load the data from the path to train yolo
    logger.info("Starting model training...")
    model.train(
        data=conf.training.data.path,
        epochs=conf.training.epochs,
        imgsz=conf.training.data.imgsz,
        device=conf.training.device,
        batch=conf.training.data.bsz,
        workers=conf.training.workers,
        project=conf.project.name,
        optimizer=conf.training.lr.optimizer,
        lr0=conf.training.lr.rate,
        momentum=conf.training.lr.gamma,
    )
    logger.info("Model training completed.")

    # Evaluate the model
    logger.info("Starting model evaluation...")
    model.val(
        data=conf.validation.data.path,
        batch=conf.validation.data.bsz,
        imgsz=conf.validation.data.imgsz,
        project=conf.project.name,
        device=conf.validation.device
    )
    logger.info("Model evaluation completed.")

    # Export the model
    logger.info(f"Exporting model in format: {conf.export_format}")
    model.export(format=conf.export_format)
    logger.info("Model export completed.")

def main() -> None:
    """
    ### Description
        Main function for the License Plate Detection model training pipeline.
    ### Parameters
        None
    ### Returns
        None
    """
    load_dotenv()
    logger.info("Environment variables loaded.")

    # Parse the arguments
    parser = argparse.ArgumentParser(description='License Plate Detection model training pipeline.')
    parser.add_argument(
        '--config', 
        type=str,
        default='config/config-dev.yaml',
        help='Path to the configuration file.'
    )
    args = parser.parse_args()

    logger.info(f"Loading configuration from: {args.config}")
    # Load the configuration
    conf = Config.from_yaml(args.config)
    logger.info("Configuration loaded successfully.")

    # Run the training pipeline
    logger.info("Starting the training pipeline...")
    run(conf)
    logger.info("Training pipeline completed.")
