# EPU_denoise


## set up environment
*if you are using `conda`, replace `mamaba` with `conda`*
```sh
mamba env create -f environment.yml
mamba activate EPU_denoise
```


## set up OpenAI API key
```sh
python src/create_env.py --openai_api_key {you api key}
```


## run the main script
```sh
python src/main.py
```
