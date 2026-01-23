#!/bin/bash
WORK_DIR="/home/onyxia/work"
CLONE_DIR="${WORK_DIR}/cours-nouvelles-donnees-site"

# Clone course repository
REPO_URL="https://github.com/inseefrlab/cours-nouvelles-donnees-site/"
git clone --depth 1 $REPO_URL $CLONE_DIR          # For prod - uncomment when prod - comment when dev
# git clone $REPO_URL $CLONE_DIR --branch dev_nt  # For dev - comment when prod - uncomment when dev

# Copy relevant notebooks to work directory
cp ${CLONE_DIR}/applications/nowcasting/{twitter.ipynb,pyproject.toml,uv.lock} ${WORK_DIR}

# Download data
mkdir data
curl "https://minio.lab.sspcloud.fr/projet-funathon/2022/diffusion/2022/Sujet%202/climate_id.txt.00" --output "data/climate_id.txt.00"  --retry 3 --retry-all-errors --max-time 5

# Remove repo
rm -rf "${CLONE_DIR}"

# Creating a Jupyter Kernel with all dependancies pre installed
# https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-within-a-project

# Install ipykernel and create a Jupyter kernel with dependencies open
uv add --dev ipykernel
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=application_nowcasting

# Delete config files
rm pyproject.toml uv.lock

