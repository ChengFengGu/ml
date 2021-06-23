#!/bin/bash


pip install jupyter notebook
pip install ipykernel
jupyter notebook --generate-config
echo "c.NotebookApp.ip='*'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.token='0305'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.open_browser=False" >> ~/.jupyter/jupyter_notebook_config.py




pip install autopep8
pip uninstall jupyter_contrib_nbextensions

echo "============================================开始环境部署======================================="
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
pip install --user jupyter_nbextensions_configurator
jupyter nbextensions_configurator enable --user



echo "=============================================环境部署完毕======================================="
jupyter notebook

  
