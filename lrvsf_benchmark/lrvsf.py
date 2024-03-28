import logging
import os
from enum import IntEnum

import pandas as pd
import torch
import yaml
from pyarrow import dataset as pds
from tqdm import tqdm

from .abc_model import LRVSFModel
from .utils import FusedRecordBatch, catchtime, human_format

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LRVSF_CATEGORIES = IntEnum(
    "LRVSF_CATEGORIES",
    [
        "Bags",
        "Feet",
        "Hands",
        "Head",
        "Lower Body",
        "Neck",
        "Outwear",
        "Upper Body",
        "Waist",
        "Whole Body",
    ],
)


class LRVSF:
    # Benchmark Properties
    K_EVALUATIONS = [0, 4, 9, 19, 49]
    DIST_EVALUATIONS = [10_000, 100_000, 500_000, 1_000_000]
    NFOLDS = 20

    def __init__(
        self,
        # Input data configuration
        data_root: str,
        conditioning: str,
        dist_img_col: str = "jpg",
        # Data Loading config
        aggregate_batches: int = 100_000,
        fragments_readahead: int = 2,
        # Misc
        dev_run: bool = False,
    ):
        """
        Parameters
        ----------
        data_root: str
            The root directory of the dataset. It should contain "products.parquet", and a "distractors" folder.
        conditioning: str
            The type of conditioning to use. Either "text" or "category".
        dist_img_col: str
            The name of the column containing the images in the parquet files.
        aggregate_batches: int
            The minimal number of sample to load in memory before calling encode_target for the distractors.
            A low value will increase the number of calls to encode_target, but will reduce the memory footprint.
        fragments_readahead: int
            The number of fragments to read ahead. A high value will increase the memory footprint, but may optimize IO.
        dev_run: bool
            If True, the benchmark should be run on a small subset of the data for debugging purposes.
        """

        self.data_sanity_check(data_root)

        self.data_root = data_root

        self.dist_img_col = dist_img_col

        conditioning_cols = {"text": "blip2_caption1", "category": "CATEGORY"}
        self.conditioning_col = conditioning_cols[conditioning]
        self.conditioning_type = conditioning

        # Data loading config
        self.aggregate_batches = aggregate_batches
        self.fragments_readahead = fragments_readahead

        # Bootstrapped Indices
        self.bootstrapped_indices = torch.load("bootstrapped_indices.pt")
        assert self.bootstrapped_indices.shape[0] == self.NFOLDS

        self.dev_run = dev_run

    @staticmethod
    def data_sanity_check(data_root: str):
        """
        Check that the data root contains the necessary files and folders : products.parquet, distractors/, ...
        """
        assert os.path.isdir(data_root), f"Data root {data_root} does not exist."

        products = os.path.join(data_root, "products.parquet")
        assert os.path.exists(
            products
        ), f"Data root {data_root} does not contain products."

        dist_folder = os.path.join(data_root, "distractors")
        assert os.path.isdir(
            dist_folder
        ), f"Data root {data_root} does not contain the distractors/ folder."

    def get_products_embeddings(
        self, model: LRVSFModel
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with catchtime("Reading products."):
            products_df = pd.read_parquet(
                os.path.join(self.data_root, "products.parquet")
            )

        simple_imgs = products_df["SIMPLE_IMG"]
        complex_imgs = products_df["COMPLEX_IMG"]

        conditioning = products_df[self.conditioning_col].tolist()

        with catchtime("Encoding queries."):
            q_product_embs = model.encode_queries(complex_imgs, conditioning)
        with catchtime("Encoding targets."):
            t_product_embs = model.encode_targets(simple_imgs)

        return q_product_embs.cpu(), t_product_embs.cpu()

    def get_distractors_embeddings(self, model: LRVSFModel) -> torch.Tensor:
        dist_folder = os.path.join(self.data_root, "distractors")
        dist_accumulator = []

        # Pyarrow Dataset for lazy loading
        dist_ds = pds.dataset(dist_folder, format="parquet")
        dist_batches = dist_ds.to_batches(
            columns=[self.dist_img_col],
            fragment_readahead=self.fragments_readahead,
            batch_size=5000,  # Enforce a batch limit for large parquet files
        )
        # If we have small parquet files, we want to fuse them until reaching at least _aggregate_batches
        # because spawning dataloader workers is costly.
        dist_batches = FusedRecordBatch(dist_batches, self.aggregate_batches)

        for i, batch in enumerate(tqdm(dist_batches)):
            # Convert to pandas - Keep pyarrow dtype to avoid copies
            dist_imgs = batch[self.dist_img_col].to_pandas(types_mapper=pd.ArrowDtype)

            with catchtime(f"Encoding {len(dist_imgs)}."):
                dist_embs = model.encode_targets(dist_imgs)

            dist_accumulator.append(dist_embs.cpu())

            if self.dev_run:
                break

        return torch.cat(dist_accumulator, dim=0)

    def evaluate_1_fold(self, model, q_embs, t_embs, dist_embs, weights=None):
        gallery = torch.cat([t_embs, dist_embs], dim=0)

        topk = model.topk(q_embs, gallery, 100)

        target = torch.arange(q_embs.shape[0])
        hits = target[:, None] == topk  # Find position of target in each line
        qwise_recall = hits.float().cumsum(dim=1)  # If R@K is 1, then R@K+1 is 1 too
        qwise_recall = (qwise_recall > 0).float()  # Against duplicates in topk

        if weights is not None:
            # Some products were de-duplicated but originally sampled multiple times
            # qwise_recall is 0 or 1 and we do a column-wise mean at the end so we can weight the rows
            qwise_recall *= weights[:, None]
            N = weights.sum()
        else:
            N = q_embs.shape[0]

        mean_rk = qwise_recall.sum(dim=0) / N

        return mean_rk

    def get_recall_nfolds(
        self, model, q_product_embs, t_product_embs, dist_embs, n_dists
    ):
        rks = []
        for i in range(self.NFOLDS):
            # Handle Query/Target Bootstrapping (pre-computed sampling with replacement)
            # We don't want duplicates in the gallery (false negatives) so we use unique and re-weight the results.
            cur_boot_indices = self.bootstrapped_indices[i, :]
            unique_indices, counts = torch.unique(cur_boot_indices, return_counts=True)

            # Sample with replacement
            rand_dist_indices = torch.randint(0, dist_embs.shape[0], (n_dists,))

            r = self.evaluate_1_fold(
                model,
                q_product_embs[unique_indices],
                t_product_embs[unique_indices],
                dist_embs[rand_dist_indices],
                weights=counts,
            )
            rks.append(r)
        rks = torch.stack(rks, dim=0)

        return rks.mean(dim=0), rks.std(dim=0, correction=1)

    def make_yaml(self, model_name, dists_shape, recalls, output_folder):
        n_dists, dim = dists_shape
        output_data = {
            "tags": ["lrvsf-benchmark"],
            "model-index": [
                {
                    "name": model_name,
                    "results": [
                        {
                            "task": {"type": "Retrieval"},
                            "dataset": {
                                "type": "Slep/LAION-RVS-Fashion",
                                "name": "LAION - Referred Visual Search - Fashion",
                                "split": "test",
                            },
                            "metrics": [
                                {
                                    "type": f"recall_at_{K+1}|{D}",
                                    "value": f"{v[0][K]*100:02.2f} ± {v[1][K]*100:.2f}",
                                    "name": f"R@{K+1} +{human_format(D)} Dist.",
                                }
                                for D, v in recalls.items()
                                for K in self.K_EVALUATIONS
                            ]
                            + [
                                {
                                    "type": "n_dists",
                                    "name": "Available Dists.",
                                    "value": n_dists,
                                },
                                {
                                    "type": "embedding_dim",
                                    "name": "Embedding Dimension",
                                    "value": dim,
                                },
                                {
                                    "type": "conditioning",
                                    "name": "Conditioning",
                                    "value": self.conditioning_type,
                                },
                            ],
                            "source": {
                                "name": "LRVSF Leaderboard",
                                "url": "https://huggingface.co/spaces/Slep/LRVSF-Leaderboard",
                            },
                        }
                    ],
                }
            ],
        }

        sanitized_model_name = model_name.replace("/", "_")
        with open(
            os.path.join(output_folder, sanitized_model_name + ".yaml"), "w"
        ) as f:
            yaml.dump(output_data, f, allow_unicode=True)

    def run(self, model: LRVSFModel, model_name: str, output_folder: str = "."):
        with torch.no_grad():
            logger.info("Encoding products.")
            q_product_embs, t_product_embs = self.get_products_embeddings(model)

            logger.info("Encoding distractors.")
            dist_embs = self.get_distractors_embeddings(model)

        if dist_embs.shape[0] < 1_000_000:
            logger.warning(
                f"Only {dist_embs.shape[0]} distractors available. This benchmark is intended for at least 1M."
            )

        recalls = {}
        with torch.no_grad():
            for n_dists in self.DIST_EVALUATIONS:
                logger.info("Starting KFold evaluation.")

                mean_r, std_r = self.get_recall_nfolds(
                    model,
                    q_product_embs,
                    t_product_embs,
                    dist_embs,
                    n_dists=n_dists,
                )

                recalls[n_dists] = [mean_r.tolist(), std_r.tolist()]

        self.make_yaml(model_name, dist_embs.shape, recalls, output_folder)
