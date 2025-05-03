# UCAS Natural Language Processing Homework 2

This is the solution of the second homework of 2025 spring UCAS Natural Language Processing By TR.

## Requirements

```bash
pip install -r requirements.txt
```

## Workflow

```bash
mkdir data
```

Download 'ChineseCorpus199801.txt' and put it in the 'data' directory.

```bash
python 0_data_clean.py
python 1_train.py FNN
python 1_train.py RNN
python 1_train.py LSTM
```