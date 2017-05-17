#!/bin/bash
printf "Initializing Setup \n"
printf "Decompressing the Google News Model... \n"
gzip -d GoogleNews-vectors-negative300.bin.gz
printf "Installing NLTK dependencies... \n"
python -m nltk.downloader -d /usr/share/nltk_data averaged_perceptron_tagger
