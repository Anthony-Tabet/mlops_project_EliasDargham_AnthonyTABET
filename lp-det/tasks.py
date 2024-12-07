from invoke import task

@task
def lint(c):
    """Run Ruff to lint the code."""
    c.run("poetry run ruff check .")

@task
def format(c):
    """Run Ruff to format the code."""
    c.run("poetry run ruff check . --fix")

@task
def type_check(c):
    """Run Mypy for type checking."""
    c.run("poetry run mypy src/")

@task
def test(c):
    """Run unit tests using pytest."""
    c.run("poetry run pytest tests/")

@task
def train(c):
    """Run the training pipeline."""
    c.run("poetry run python src/detector_training/train.py --config config/config-dev.yaml")

@task
def generate_docs(c):
    """Generate project documentation using pdoc."""
    c.run("poetry run pdoc --html --output-dir docs src")