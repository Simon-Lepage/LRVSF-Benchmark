<h1 align="center">LAION RVS Fashion - Benchmark</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2306.02928">
        <img alt="GitHub release" src="https://img.shields.io/badge/arXiv-2306.02928-b31b1b.svg">
    </a>
    <img alt="Apache License" src="https://img.shields.io/badge/License-Apache_2.0-green.svg">
</p>

<h4 align="center">
    <p>
        <a href="https://huggingface.co/spaces/Slep/LRVSF-Leaderboard">Leaderboard</a> |
        <a href="#installation">Installation</a> |
        <a href="#usage">Usage</a> |
        <a href="#citing">Citing</a>
    <p>
</h4>

# Installation

1. Install the `lrvsf_benchmark` library from this repository. 

```sh
pip install git+https://github.com/Simon-Lepage/LRVSF-Benchmark.git
```

2. Install any other library you might need (for example your favorite versions of `transformers`, `torchvision`, ...).

# Usage

## Dataset Preparation 📁

We will provide a version of the test set on [zenodo.org](https://zenodo.org/) soon. It will contain : 
- `products.parquet` : query and target images for each product, alongside with categorical and textual conditioning. The images were downloaded with img2dataset, and are stored in JPG format with shortest side resized to 256, keeping the original aspect ratio.
- `distractors_urls.parquet` : URLs of the 2M test distractors. We also provide a script using [img2dataset](https://github.com/rom1504/img2dataset) to easily download the pictures.

The evaluation code expects this data to be structured in the following way, with distractors containing as many `parquet` files as you see fit. We recommend 10k-100k distractors per parquet file. Avoid too large files, as they will be entirely loaded into memory.
```
data/
└ products.parquet
└ distractors/
    └ *.parquet
    └ ...
```

## Model Evaluation ⚙️

### 1. Wrap your model

Models should implement the following interface, defining `encode_queries`, `encode_targets` and `topk`.

The encoding functions receive lists of items gathered from the parquet files (list of `PIL.Image` and `str`). It is your responsibility to define your own DataLoader to apply any preprocessing steps. These functions should return the embeddings as a `torch.Tensor`. You can use `torch.nn.DataParallel` to use multiple GPUs.

The `topk` function will be called with the embeddings produced previously. For each query embedding in `q_emb`, it must return the top-k indices sorting the target embeddings `t_embs` by similarity. A simple implementation could use `faiss` or `torch.topk(...).indices`.

```python
class MyModel(lrvsf_benchmark.LRVSFModel):
    def __init__(self, ...):
        super().__init__()
        pass

    def encode_queries(self, imgs, conds):
        pass

    def encode_targets(self, imgs):
        pass

    def topk(self, q_embs, t_embs, topk):
        pass
```

### 2. Evaluate your model

Pass this model to `lrvsf_benchmark.LRVSF` class. It will generate the embeddings, and repeatedly call `topk` on bootstrapped subsets to compute the results. 

```python
evaluator = LRVSF(...)
```
**Mandatory arguments:**
- `data_root` : the root of the test set. `data/` in the previous example.
- `conditioning` : either `category` or `text`.

**Optional arguments:**
- `dist_img_col` (str): the name of the column containing the images in your parquet files. *Default: "jpg"*
- `aggregate_batches` (int): controls the number of samples to aggregate before calling the encoding methods. Useful if you have small parquet files, as spawning a DataLoader for few images is inefficient. *Default: 100_000*
- `fragments_readahead` (int): Passed to `pyarrow.dataset.dataset`. Controls the number of parquet files loaded in memory at the same time. Small for large parquet files, large for small parquet files. *Default: 2*
- `dev_run` (bool): If set to True, will run the evaluation on a subset of the distractors. *Default: False*

Finally, run the evaluator to compute the metrics and export them as YAML: 
```python
evaluator.run(my_model, "output_filename.yaml")
```

### 3. Update your model card

Copy the content of the produced YAML output to the header of your `README.md` file. You should see the metrics appear in the model card. Refresh the [Leaderboard](https://huggingface.co/spaces/Slep/LRVSF-Leaderboard).

> Please refer to `examples/condvit_b16_cat.py` for an example of evaluation script. You can also look at the [README.md file of our CondViT-B16](https://huggingface.co/Slep/CondViT-B16-cat/blob/main/README.md) for a YAML formatting reference.

# Citing

LAION-RVS-Fashion was introduced in [Weakly-Supervised Conditional Embedding for Referred Visual Search](https://arxiv.org/abs/2306.02928). If this work is helpful for your research, please cite : 

```bibtex
@article{lepage2023condvit,
  title={Weakly-Supervised Conditional Embedding for Referred Visual Search},
  author={Lepage, Simon and Mary, Jérémie and Picard, David},
  journal={arXiv:2306.02928},
  year={2023}
}
```