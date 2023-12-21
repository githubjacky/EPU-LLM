.PHONY: tests docs

# installing dependencies
dependencies:
	poetry install
	# poetry run pre-commit install

# activate virtual environment
env: dependencies
	poetry shell

# tests:
# 	pytest
#
# docs:
# 	@echo Save documentation to docs... 
# 	pdoc src -o docs --force
# 	@echo View API documentation... 
# 	pdoc src --http localhost:8080	

predict:
	poetry run python src/models/predict.py

finetune:
	poetry run python src/models/fine_tune.py

mlflow:
	poetry run mlflow ui -h 0.0.0.0 -p 5050

jupyter:
	poetry run jupyter lab --ip=0.0.0.0 --port=8888


# docker
dpredict:
	docker compose run --rm llm-predict

dfinetune:
	docker compose run --rm fine-tune

dmlflow:
	docker compose run --rm --service-ports mlflow-ui

djupyter:
	docker compose run --rm --service-ports jupyter-lab
