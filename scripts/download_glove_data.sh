#!/usr/bin/env bash
rm data/glove*
curl -o data/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
cd data
unzip glove.6B.zip
cd ..
