# EPU_denoise


## set up environment
*if you are using `conda`, replace `mamaba` with `conda`*
```sh
mamba env create -f environment.yml
mamba activate EPU_denoise
```

## download data from BigKinds 
*arguments has the restricted format*
- `p`: press
- `b`: begin date
- `e`: end date
- `i`: number of days to query eachtime
- default configuration:
```{bash}
./scrap -p 매일경제신문_한국경제 -b 2023-07-26 -e 2023-07-29 -i 2
```


## set up OpenAI API key
```sh
python src/create_env.py --openai_api_key {your api key}
```


## run the main script
```sh
python src/main.py
```
