source env/.env

docker run --rm \
	--name chatgpt_predict \
	-e OPENAI_API_KEY=$OPENAI_API_KEY \
	-v $PWD/data:/epu_denoise/data \
	-v $PWD/prompt_template/:/epu_denoise/prompt_template \
	-v $PWD/src/models:/epu_denoise/src/models \
	-v $PWD/config:/epu_denoise/config \
	-v $PWD/mlruns:/epu_denoise/mlruns \
	-v $PWD/log:/epu_denoise/log \
	0jacky/epu_denoise:model \
	python src/models/predict.py
