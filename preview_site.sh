# cd cours-nouvelles-donnees-site/
# git checkout maj-25
uv sync
source rv_installation.sh
rv sync
uv run bash download_nlp_reqs.sh
uv run quarto preview
