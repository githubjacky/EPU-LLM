#!/bin/sh
source .env

curl https://api.openai.com/v1/fine_tuning/jobs?limit=2 \
	-H "Authorization: Bearer $OPENAI_API_KEY"
