if ! command -v python3 &> /dev/null; then echo "python3 missing"; exit 1; fi
if ! command -v pip &> /dev/null; then echo "pip missing"; exit 1; fi

python3 -m pip install --upgrade pip

# install formatter
pip install black > /dev/null

# dev: find out requirements
rm -rf requirements.txt
pip install pipreqs
pipreqs .

# prod: install requirements
pip install -r requirements.txt
