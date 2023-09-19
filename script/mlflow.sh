DOCKER_USER=r12323011

docker run --rm \
	--name mlflow_ui \
	-v $PWD/mlruns:/home/$DOCKER_USER/epu_denoise/mlruns \
	-p 5050:5050 \
	0jacky/epu_denoise:model \
	mlflow server -h 0.0.0.0 -p 5050
