# UCAS Natural Language Processing Homework 3

This is the solution of the third homework of 2025 spring UCAS Natural Language Processing By TR.

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
python data_clean.py
python experiment.py --config_dir ./configs
python get_results.py
```