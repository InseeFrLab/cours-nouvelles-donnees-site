#!/bin/bash
WORK_DIR="/home/onyxia/work"
CLONE_DIR="${WORK_DIR}/cours-nouvelles-donnees-site"

# Clone course repository
REPO_URL="https://github.com/inseefrlab/cours-nouvelles-donnees-site/"
git clone --depth 1 $REPO_URL $CLONE_DIR

# Copy relevant notebooks to work directory
cp ${CLONE_DIR}/applications/images/{classification_oiseau.ipynb,donnees_satellite.ipynb, pyproject.toml, uv.lock} ${WORK_DIR}

# Remove repo
rm -rf "${CLONE_DIR}"

# Creating a Jupyter Kernel with all dependancies pre installed
# https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-within-a-project

# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install ipykernel and create a Jupyter kernel
uv add -dev ipykernel
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=project

# Open the first notebook when starting Jupyter Lab
# Generate Jupyter config if it doesn't exist
if [ ! -f /home/onyxia/.jupyter/jupyter_server_config.py ]; then
    uv run jupyter server --generate-config -y
fi
echo "c.LabApp.default_url = '/lab/tree/classification_oiseau.ipynb'" >> /home/onyxia/.jupyter/jupyter_server_config.py

# Run Jupyter Lab
uv run --with jupyter jupyter lab


