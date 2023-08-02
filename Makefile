.PHONY: tests docs

dependencies: 
	@echo "Initializing Git..."
	git init
	@echo "Installing dependencies..."
	poetry install
	poetry run pre-commit install

env: dependencies
	@echo "Activating virtual environment..."
	poetry shell

download:
	poetry run python src/data/download_bigkinds_data.py

predict:
	poetry run python src/predict.py

tests:
	pytest

docs:
	@echo Save documentation to docs... 
	pdoc src -o docs --force
	@echo View API documentation... 
	pdoc src --http localhost:8080	
