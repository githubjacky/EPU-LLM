# epu_denoise

## Set up the virtual environment
There are two ways for setting puthe virtual environment. One is
[poetry](https://github.com/python-poetry/poetry), and the other is [docker](https://github.com/docker). For the detailed information on how to download and make it works, pleas check out the link.
- docker: Dockerfile is under `dockerfile/`
- install repo dependencies using poetry: `make dependencies`


## uee LLM to make prediction
1. change the main configurations in `config/main.yaml`
2. modify the model configuration file in *config/model/{model}.yaml*
    - now there is only model available, which is the chatgpt
    - default: gpt-3.5-turbo-1106
3. run the command
```bash
# poetry venv
make predict

# docker
make dpredict
```

## fine-tune LLM
1. change the main configurations in `config/main.yaml`
2. modify the model configuration file in `config/model/{model}.yaml`
    - now there is only model available, which is the chatgpt
3. run the command
```bash
# poetry venv
make finetune

# docker
make dfinetune
```

There are more functionalites provided by this repo such as the mlflow-ui, or
jupyter lab editor, please check out the Makefile.
```bash
# poetry venv(mlflow-ui, jupyter lab)
make mlflow
make jupyter

# docker(mlflow-ui, jupyter lab)
make dmlflow
make djupyter
```
