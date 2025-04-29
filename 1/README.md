# UCAS Natural Language Processing Homework 1

This is the solution of the first homework of 2025 spring UCAS Natural Language Processing By TR.

## Files

- `arxivpapers.py`: Python script to download and parse arXiv papers from [arXiv](https://arxiv.org/).
- `ppnews.py`: Python script to download and parse People's Daily News articles from [PDN](http://paper.people.com.cn/rmrb/pc/layout/202503/19/node_01.htmll).
- `cleaner_chs.py`: Python script to clean Chinese text.
- `cleaner_eng.py`: Python script to clean English text.
- `experiment.py`: Python script to do experiments.
- `data_exploer.ipynb`: Jupyter notebook to plot the data.

## Requirements

```bash
pip install -r requirements.txt
```

## Workflow

1. Create a `data` directory under the project root directory.

2. Get the raw text data from arXiv and PDN.

```bash
python arxivpapers.py
python ppnews.py
```

3. Clean the text data.

```bash
python cleaner_chs.py
python cleaner_eng.py
```

4. Calculate the entropy of the dataset.

For example: 

```bash
python experiments.py data/cleaned_eng.txt --method accumulate
python experiments.py data/cleaned_eng.txt --method piecewise
```

5. Plot the data using `data_exploer.ipynb`.

