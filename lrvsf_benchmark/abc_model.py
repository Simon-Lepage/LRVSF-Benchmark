from abc import ABC, abstractmethod

import torch
from PIL.Image import Image


class LRVSFModel(ABC, torch.nn.Module):
    @abstractmethod
    def encode_queries(self, imgs: list[Image], conds: list[str]) -> torch.Tensor:
        """
        Returns a list of conditional embeddings for each of the query images.

        Parameters
        ----------
        imgs: list of PIL.Image
            A list of images to encode
        conds: list of str
            A list of conditions for each image

        :return: A tensor of shape len(imgs) x D containing the embeddings for each image.
        """
        pass

    @abstractmethod
    def encode_targets(self, imgs: list[Image]) -> torch.Tensor:
        """
        Returns a list of embeddings for the given target images.

        Parameters
        ----------
        imgs: list of PIL.Image
            A list of images to encode

        :return: A tensor of shape len(imgs) x D containing the embeddings for each image.
        """
        pass

    @abstractmethod
    def topk(
        self, q_embs: torch.Tensor, t_embs: torch.Tensor, topk: int
    ) -> torch.Tensor:
        """
        Returns the indices of topk most similar targets for each query. Returned indices should be unique on each line.

        Parameters
        ----------
        q_embs: torch.Tensor
            The embeddings of the queries

        t_embs: torch.Tensor
            The embeddings of the targets

        topk: int

        :return: An array of shape len(q_embs) x topk containing the indices of the topk most similar targets for each query.
        """
        pass
