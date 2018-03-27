!/usr/bin/env bash
rm data/glove*
curl -o data/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
cd data
unzip glove.6B.zip
cd ..
python scripts/convert_glove2numpy.py --path data/glove.6B.300d.txt
