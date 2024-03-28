import os
from io import BytesIO

import torch
import torch.distributed
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from lrvsf_benchmark import LRVSF, LRVSFModel


class DS:
    def __init__(self, prepro, images, conditioning=None):
        if conditioning is not None:
            assert len(images) == len(conditioning)

        self.prepro = prepro
        self.images = images
        self.conditioning = conditioning

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(BytesIO(self.images.iloc[i])).convert("RGB")

        if self.conditioning is not None:
            b = self.prepro(images=img, categories=self.conditioning[i])
        else:
            b = self.prepro(images=img)

        return b


class CondViTModel(LRVSFModel):
    def __init__(self, **dl_kwargs):
        super().__init__()

        params = {
            "pretrained_model_name_or_path": "Slep/CondViT-B16-cat",
            "trust_remote_code": True,
        }

        self.model = AutoModel.from_pretrained(**params)
        self.processor = AutoProcessor.from_pretrained(**params)

        self.model = self.model.cuda()
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model.eval()

        self.dl_kwargs = dl_kwargs

    def _unified_encode(self, imgs, conds=None):
        ds = DS(self.processor, imgs, conditioning=conds)
        dl = torch.utils.data.DataLoader(
            ds, shuffle=False, drop_last=False, **self.dl_kwargs
        )

        embs = []
        for b in tqdm(dl):
            with torch.no_grad():
                b.to("cuda")
                e = self.model(**b)
                embs.append(e.cpu())
        embs = torch.cat(embs, dim=0)
        embs = torch.nn.functional.normalize(embs, dim=1)

        return embs

    def encode_queries(self, imgs, conds):
        return self._unified_encode(imgs, conds)

    def encode_targets(self, imgs):
        return self._unified_encode(imgs)

    def topk(self, q_embs, t_embs, topk):
        q_embs, t_embs = q_embs.cuda(), t_embs.cuda()

        similarities = q_embs @ t_embs.T
        i = torch.topk(similarities, topk, dim=1).indices

        return i.cpu()


if __name__ == "__main__":
    condvit = CondViTModel(
        num_workers=len(os.sched_getaffinity(0)),
        batch_size=512,
    )

    evaluator = LRVSF(
        data_root="/tmp/lrvsf-testset",
        conditioning="category",
    )

    evaluator.run(condvit, model_name="CondViT-B16-cat")
