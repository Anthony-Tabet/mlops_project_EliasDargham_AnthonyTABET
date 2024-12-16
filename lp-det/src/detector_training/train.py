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
import onnx
import mlflow
from ultralytics import settings
import numpy as np
from ultralytics import YOLO
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
    logger.info("Turning off ultralytics MLflow logging. Will use global MLflow logging instead.")
    settings.update({"mlflow": False})

    logger.info(f"Setting up MLflow experiment: {conf.project.name}")
    mlflow.set_experiment(conf.project.name)
    with mlflow.start_run():
        model = YOLO(model=f'{conf.training.model}{conf.training.size}.pt')

        model.model_name = conf.project.name

        logger.info("Model loaded and configured.")

        # Load the data from the path to train yolo
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
            momentum=conf.training.lr.gamma
        )
        logger.info("Model training completed.")
        # session_id = session.info.run_id
        # session_name = session.info.run_name
        # print(f"MLflow session ID: {session_id}")
        # print(f'Sessoion Name: {session_name}')

        # Log the configuration to mlflow
        mlflow.log_params(conf.to_dict())

        # Log the results to mlflow
        print(model.metrics.results_dict)
        new_results_dict = {
            key.replace('metrics/', '').replace('(B)', ''): value\
                for key, value in model.metrics.results_dict.items()
        }
        mlflow.log_metrics(new_results_dict)

        logger.info(f"Exporting model in format: {conf.export_format}")
        onnx_model_path = model.export(format=conf.export_format, opset=20)
        logger.info("Model export completed.")

        # Save the model with mlflow
        onnx_model = onnx.load(onnx_model_path)
        # input exaple
        input_example = np.random.randn(1, 3, conf.training.data.imgsz, conf.training.data.imgsz)
        mlflow.onnx.log_model(onnx_model, artifact_path="model", input_example=input_example)
        logger.info("Model registered in MLflow.")

def main() -> None:
    """
    ### Description
        Main function for the License Plate Detection model training pipeline.
    ### Parameters
        None
    ### Returns
        None
    """
    load_dotenv('.env')
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
