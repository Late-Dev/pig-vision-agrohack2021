python3 -m venv venv/
source venv/bin/activate
pip install gdown

rm -rf models
gdown --id 1Ix2a5chTQ6KTo4XXE0JAjfYLDi1XUC4o -O ./
unzip models.zip
rm -f models.zip*