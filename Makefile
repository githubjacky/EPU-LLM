.PHONY: dependencies env build reason predict compile finetune mlflow jupyter dreason dpredict dcompile dfinetune dmlflow djupyter

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

reason:
	poetry run python script/reason.py

compile:
	poetry run python script/compile_prompt.py

predict:
	poetry run python script/predict.py


finetune:
	poetry run python script/fine_tune.py

mlflow:
	poetry run mlflow ui -h 0.0.0.0 -p 5050

jupyter:
	poetry run jupyter lab --ip=0.0.0.0 --port=8888


# docker
build:
	docker compose build

dreason:
	docker compose run --rm llm-reason

dcompile:
	docker compose run --rm compile-prompt

dpredict:
	docker compose run --rm llm-predict


dfinetune:
	docker compose run --rm fine-tune

dmlflow:
	docker compose run --rm --service-ports mlflow-ui

djupyter:
	docker compose run --rm --service-ports jupyter-lab
