# License Plate Detection Project: Elias Dargham & Anthony Tabet

## Introduction
Welcome to our License Plate Detection project. This project uses machine learning and computer vision to accurately detect and read license plates from images and video feeds. We leverage robust MLOps practices to ensure a scalable and maintainable system.

## Project Structure
- **lp-det**: Configuration files for Prometheus and Alertmanager for monitoring.
- **detector_training**: Training scripts for the license plate detection models.
- **inference_tracking**: Modules for tracking inference metrics and real-time data processing.
- **tests**: Test suite for ensuring the functionality and stability of our models and infrastructure.

## System Requirements
- Docker & Docker Compose
- Python 3.8 or higher
- Poetry for Python dependency management

## Dataset
This project uses a publicly available dataset which can be found at the following link:
[License Plate Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)
Please ensure you comply with the dataset's usage policy before utilizing it for training or testing.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourrepo/MLOPS_PROJECT_ELIASDARGHAM_ANTHONTABET.git
cd MLOPS_PROJECT_ELIASDARGHAM_ANTHONTABET
```

### 2. Install Dependencies
we use Poetry to manage Python Dependencies. Install Them by Running:
```bash
poetry install
```

### 3. Environment Setup
Configure your environment variables appropriately:
```bash
cp .env.example .env
```

### 4. Build and Run with Docker
```bash
docker-compose up --build
```

## Accessing Monitoring Tools:
- **Prometheus:** Access at http://localhost:9090 for system monitoring.
- **Alertmanager:** Access at http://localhost:9093 for alert management

## Testing:
Run the automated tests to check system integrity:
```bash
docker-compose exec your_service_name poetry run pytest 
```

## Contributing
We encourage contributions! Please follow these steps:

1. Fork the repo.
2. Create your feature branch (git checkout -b my-new-feature).
3. Commit your changes (git commit -am 'Add some feature').
4. Push to the branch (git push origin my-new-feature).
5. Create a new Pull Request.