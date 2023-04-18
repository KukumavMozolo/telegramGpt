<!-- Just fill in the brackets -->
# telegramGpt
Replace your friends with machines, easily train llama on you telegram data 

Based on https://github.com/tloen/alpaca-lora

[![release](https://img.shields.io/badge/release-v0.0-red.svg?style=flat-square)]()


Installation
-----------
Requirements:

    python3.10
    pipenv

Install deps:

    pipenv install --ignore-pipfile


Features
--------

- memory efficient training using https://github.com/TimDettmers/bitsandbytes
- Gradio Interface for inference


Training
--------
copy your telegram data message*.html's into data/your-dataset-name

start training with

    python -d your-dataset-name -e 1 --create_dataset

Specify modelname with -m modelname for models other than decapoda-research/llama-7b-hf

resume from checkpoint via -re path/to/checkpoint

number of epochs -e float

--create_dataset to create the dataset at first run or --no-create_dataset to use the old dataset from previous run
