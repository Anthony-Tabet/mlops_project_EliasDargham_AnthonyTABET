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

import onnx
import mlflow
from ultralytics import YOLO

from detector_training.config_loader import Config


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
    with mlflow.start_run():
        model = YOLO(model=f'{conf.training.model}{conf.training.size}.pt')
        model.model_name = conf.project.name

        # Load the data from the path to train yolo
        results = model.train(
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
        # session_id = session.info.run_id
        # session_name = session.info.run_name
        # print(f"MLflow session ID: {session_id}")
        # print(f'Sessoion Name: {session_name}')
        
        # Log the configuration to mlflow
        mlflow.log_params(conf.to_dict())
        

        # Log the results to mlflow
        mlflow.log_metrics(results)

        onnx_model_path = model.export(format=conf.export_format, opset=20)

        # Save the model with mlflow
        onnx_model = onnx.load(onnx_model_path)
        mlflow.onnx.log_model(onnx_model, artifact_path="model")

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
