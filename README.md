# ProtoRA

This is the PyTorch implementation of **ProtoRA: Training-Free Cross-City Region Embedding Alignment through Prototype-Relative Representations**.

## Data

`region_data/` — Data of three cities: Chengdu (CD), Beijing (BJ), and Xi'an (XA). The data of each city contains POI data, mobility data, and the socioeconomic indicator data.

## Code

- `ProtoRA.py` — Main implementation of the ProtoRA framework.

- `run.py` — A file to run the ProtoRA model. Note that there are some hyperparameters that can be set in this file.

- `evaluator_cross_city.py` — The files contain the details of cross-city region-level socioeconomic prediction tasks.

- `utils.py` — These are the utility functions for data processing and prototype construction.

## Region Embeddings

`region_embs/` — These are the multiple pretrained region embeddings. These embeddings are trained by methods: **ReCP** (Li et al., 2024), **HREP** (Zhou et al., 2023), and **HAFusion** (Sun et al., 2024).
