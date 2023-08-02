# epu_denoise

## Set up the environment and Install dependencies
1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Set up the environment:
```bash
make dependencies
```


## download the data from [bigkinds](https://www.bigkinds.or.kr/v2/news/index.do)
1. modify the configuration file in *config/porcess/scrapping.yaml*
2. run the command
```{bash}
make download
```

## uee LLM to make prediction
1. change the default model in *config/main.yaml*
    - chatgpt3.5-turbo
2. modify the configuration file in *config/model/{model.yaml}*
3. run the command
```{bash}
make predict
```
