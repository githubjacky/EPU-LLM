source env/.env
docker run --rm \
	--name jupyterlab \
	-e OPENAI_API_KEY=$OPENAI_API_KEY \
	-v $PWD/notebooks:/epu_denoise/notebooks \
	-v $PWD/data:/epu_denoise/data \
	-v $PWD/prompt_template/:/epu_denoise/prompt_template \
	-v $PWD/src/:/epu_denoise/src \
	-p 8888:8888 \
	0jacky/epu_denoise:model \
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
