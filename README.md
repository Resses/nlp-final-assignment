# Final Assignment

## Step 1
Download Google News Word2Vec model: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit. Save the file as it is suggested (GoogleNews-vectors-negative300.bin.gz) in the location of the repository.

## Step 2
Install requirements

pip install -r requirements.txt

## Step 3
Run in the command line the setup file.

./setup.sh

## Step 4
Run the main script 

For production with both dev and training
python ne.py

For running on the dev set, run:
python ne.py --dev 

For running on the dev set with CV, run:
python ne.py --dev --cv

For running on the dev set with details on errors, run:
python ne.py --dev --details

usage: ne.py [-h] [--dev] [--cv] [--details]

optional arguments:
-h, --help  show this help message and exit
--dev       Flag that determines whether to use training and dev set together or not.
--cv        Flag that determines whether to do cross validation.
--details   Flag that determines whether to show details of mismatches.
