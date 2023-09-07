docker run --rm \
	--name get_system_human_prompt_template \
	-v $PWD/prompt_template/:/epu_denoise/prompt_template \
	-v $PWD/src/models/get_system_human_prompt_template.py:/epu_denoise/src/models/get_system_human_prompt_template.py \
	0jacky/epu_denoise:model \
	python -u src/models/get_system_human_prompt_template.py
