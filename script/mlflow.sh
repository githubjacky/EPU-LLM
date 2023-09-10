docker run --rm \
	--name mlflow_ui \
	-v $PWD/mlruns/:/epu_denoise/mlruns \
	-p 5050:5050 \
	0jacky/epu_denoise:model \
	mlflow ui -h 0.0.0.0 -p 5050
