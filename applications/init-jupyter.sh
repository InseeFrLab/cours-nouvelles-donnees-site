#!/bin/bash
WORK_DIR="/home/onyxia/work"
CLONE_DIR="${WORK_DIR}/images-donnees-emergentes"

# Clone course repository
REPO_URL="https://git.lab.sspcloud.fr/tseimandi/images-donnees-emergentes.git"
git clone --depth 1 $REPO_URL $CLONE_DIR

# Copy relevant notebooks to work directory
cp ${CLONE_DIR}/{classification_oiseau.ipynb,donnees_satellite.ipynb} ${WORK_DIR}

# Remove repo and useless lost+found directory
rm -rf ${CLONE_DIR}/ ${WORK_DIR}/lost+found

# Open the first notebook when starting Jupyter Lab
jupyter server --generate-config
echo "c.LabApp.default_url = '/lab/tree/classification_oiseau.ipynb'" >> /home/onyxia/.jupyter/jupyter_server_config.py
