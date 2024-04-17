#!/bin/bash

tar -xzf migration_env.tar.gz
export PYTHONPATH=$PWD/migration_env

mkdir migration

# Run the Python script 
python3 nedwingreenospool.py $1
