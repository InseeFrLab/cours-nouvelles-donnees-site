#!/bin/bash
WORK_DIR="/home/onyxia/work"
CLONE_DIR="${WORK_DIR}/cours-nouvelles-donnees-site"

# Clone course repository
REPO_URL="https://github.com/inseefrlab/cours-nouvelles-donnees-site/"
git clone --depth 1 $REPO_URL $CLONE_DIR

# Copy relevant notebooks to work directory
cp ${CLONE_DIR}/applications/nowcasting/{twitter.ipynb,requirements.txt,setup.sh} ${WORK_DIR}

# Remove repo and useless lost+found directory
rm -rf ${CLONE_DIR}/ ${WORK_DIR}/lost+found

# Run setup and delete files
source setup.sh
rm requirements.txt setup.sh

# Open the first notebook when starting Jupyter Lab
jupyter server --generate-config
echo "c.LabApp.default_url = '/lab/tree/twitter.ipynb'" >> /home/onyxia/.jupyter/jupyter_server_config.py
