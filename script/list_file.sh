#!/bin/sh
source .env

curl https://api.openai.com/v1/files \
	-H "Authorization: Bearer $OPENAI_API_KEY"
