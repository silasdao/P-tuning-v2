import torch
from torch import Tensor
from typing import List, Dict, Union, Tuple
import numpy as np
from tqdm.autonotebook import trange

from colbert.evaluation.loaders import load_colbert
import torch.nn as nn

class Maxsim(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.batch_size = args.bs
        self.doc_maxlen = args.doc_maxlen
        self.query_maxlen = args.query_maxlen
        self.dim = args.dim

    def forward(self, Q, D):
        res = []
        q_size = Q.shape[0]
        d_size = D.shape[0]
        for start_idx in trange(0, d_size, self.batch_size):
            batch_D = D[start_idx:start_idx+self.batch_size, :, :]
            batch_res = self.score(Q, batch_D)
            res.append(batch_res)
        return torch.hstack(res)

    def score(self, Q, D, similarity_metric='cosine'):
        q_size = Q.shape[0]
        d_size = D.shape[0]
        d = D.expand(q_size, d_size, self.doc_maxlen, self.dim)
        q = Q.expand(d_size, q_size, self.query_maxlen, self.dim).permute(1,0,2,3)

        return (q @ d.permute(0, 1, 3, 2)).max(3).values.sum(2)