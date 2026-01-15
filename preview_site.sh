# cd cours-nouvelles-donnees-site/
# git checkout maj-25 
pip install -r requirements.txt
Rscript -e "renv::restore()"
bash download_nlp_reqs.sh
quarto preview