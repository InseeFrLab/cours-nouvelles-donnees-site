# cd cours-nouvelles-donnees-site/
# git checkout maj-25
source rv_installation.sh && source ~/.bashrc
rv sync
uv sync
uv run bash download_nlp_reqs.sh
uv run quarto preview
