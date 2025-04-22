#!/bin/sh

sudo apt-get install --assume-yes --no-install-recommends cmake libsndfile1
pip install --requirement requirements.txt
sagemaker-code-editor --install-extension ms-python.black-formatter --extensions-dir /opt/amazon/sagemaker/sagemaker-code-editor-server-data/extensions
