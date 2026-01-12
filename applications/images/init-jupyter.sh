#!/bin/bash
WORK_DIR="/home/onyxia/work"
CLONE_DIR="${WORK_DIR}/cours-nouvelles-donnees-site"

# Clone course repository
REPO_URL="https://github.com/inseefrlab/cours-nouvelles-donnees-site/"
git clone $REPO_URL $CLONE_DIR --branch dev_nt  # Remove dev_nt when pushed to main

# Copy relevant notebooks to work directory
cp ${CLONE_DIR}/applications/images/{classification_oiseau.ipynb,donnees_satellite.ipynb,pyproject.toml,uv.lock} ${WORK_DIR}

# Remove repo
rm -rf "${CLONE_DIR}"

# Creating a Jupyter Kernel with all dependancies pre installed
# https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-within-a-project

# Install ipykernel and create a Jupyter kernel with dependencies open
uv add --dev ipykernel
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=application_images

# Delete config files
rm pyproject.toml uv.lock

