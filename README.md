# EPU-LLM
*Utilzing LLM to construct economic textual indicators.*


## Tools used in this project
- [docker](https://www.docker.com/): runtime environment
- [Poetry](https://python-poetry.org/docs/#installation): package management
* [hydra](https://hydra.cc/): manage configuration files
* [DVC](https://dvc.org/): data version control
* [sphinx](https://www.sphinx-doc.org/en/master/): automatically create an API documentation for your project
- [mlflow](https://mlflow.org/#core-concepts): experiment tracking


## Set up the environment, and Install dependencies
1. Install [docker](https://docs.docker.com/get-docker/)
2. modify the `.env.example`, assigning the environment variables and rename it as `.env`
3. create the docker image:
```bash
make build
```
To clean up the docker image:
```sh
make clean
```


## Container Services
```sh
# unit test
make pytest

# project documentation
# after typing the command open docs/_build//html/index.html in the browser
make doc

# development IDE - Jupyter Lab
make jupyter

# MLflow tracking UI
make mlflow
```


## process scripts/notebooks
