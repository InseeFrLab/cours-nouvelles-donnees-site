#!/bin/bash
pip install -r requirements.txt

mkdir data
export MC_HOST_minio=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY:$AWS_SESSION_TOKEN@$AWS_S3_ENDPOINT
mc cp "minio/projet-funathon/diffusion/2022/Sujet 2/climate_id.txt.00" data/climate_id.txt.00
