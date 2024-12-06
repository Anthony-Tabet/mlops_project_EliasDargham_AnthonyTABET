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

from ultralytics import YOLO
import mlflow

from lp_det.config_loader import Config


def run(conf: Config) -> None:
    """
    ### Description
        Main function for the training pipeline of the License Plate Detection model.
    ### Parameters
        conf (Config): Configuration instance for the training pipeline.
    ### Returns
        None
    """
    mlflow.set_experiment(conf.project.name)
    # Load the model
    model = YOLO(model=f'{conf.training.model}{conf.training.size}.pt')
    model.model_name = 'lp-det'

    # Register the model with mlflow
    mlflow.pytorch.log_model(model, 'model')

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
        momentum=conf.training.lr.gamma,
    )

    # Evaluate the mmodel
    model.val(
        data=conf.validation.data.path,
        batch=conf.validation.data.bsz,
        imgsz=conf.validation.data.imgsz,
        project=conf.project.name,
        device=conf.validation.device
    )

    # Export the model
    model.export(format=conf.export_format)

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
    # Parse the arguments
    parser = argparse.ArgumentParser(description='License Plate Detection model training pipeline.')
    parser.add_argument(
        '--config', 
        type=str,
        default='config/config-dev.yaml',
        help='Path to the configuration file.'
    )
    args = parser.parse_args()

    # Load the configuration
    conf = Config.from_yaml(args.config)

    # Run the training pipeline
    run(conf)
