from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import torch
from torch import nn

from tqdm import tqdm

class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :return:
        """
        ranks = torch.ones(len(queries))
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _ = self.forward(these_queries)
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]   # Add the tail of this (b_begin + i) query
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks
    def get_ranking2(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            model2,
            weight1: float = 1.2, weight2: float = 0.8,  # 加权融合的权重
            batch_size: int = 1000, chunk_size: int = -1,
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :return:
        """
        ranks = torch.ones(len(queries))
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _ = self.forward(these_queries)
                    scores2, _ = model2.forward(these_queries)
                    scores = weight1 * scores + weight2 * scores2
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]   # Add the tail of this (b_begin + i) query
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks

class RESCAL(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3,tao:float=0.05
    ):
        super(RESCAL, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.tao=tao

        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], rank, sparse=True),
            nn.Embedding(sizes[1], rank * rank, sparse=True),
        ])

        nn.init.xavier_uniform_(tensor=self.embeddings[0].weight)
        nn.init.xavier_uniform_(tensor=self.embeddings[1].weight)

        self.lhs = self.embeddings[0]
        self.rel = self.embeddings[1]
        self.rhs = self.embeddings[0]

    def forward(self, x,p_hr=None,p_tr=None):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1]).reshape(-1, self.rank, self.rank)
        rhs = self.rhs(x[:, 2])
        if p_hr is not None:
            h = self.lhs(p_hr[:, 0])
            r = self.rel(p_hr[:, 1]).reshape(-1, self.rank, self.rank)
            return (torch.bmm(lhs.unsqueeze(1), rel)).squeeze(), (torch.bmm(h.unsqueeze(1), r)).squeeze()
        if p_tr is not None:
            t = self.rhs(p_tr[:, 0])
            r = self.rel(p_tr[:, 1]).reshape(-1, self.rank, self.rank)
            return (torch.bmm(rhs.unsqueeze(1), rel)).squeeze(), (torch.bmm(t.unsqueeze(1), r)).squeeze()


        return (torch.bmm(lhs.unsqueeze(1), rel)).squeeze() @ self.rhs.weight.t(),[(lhs, rel, rhs)]




class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3,tao:float=0.05
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True)
            for s in sizes[:3]
        ])

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.lhs = self.embeddings[0]
        self.rel = self.embeddings[1]
        self.rhs = self.embeddings[2]

    def forward(self, x,p_hr=None,p_tr=None):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        self_hr=lhs * rel
        self_tr=rhs * rel
        if p_hr is not None:
            h = self.lhs(p_hr[:, 0])
            r = self.rel(p_hr[:, 1])
            return self_hr, h * r
        if p_tr is not None:
            t = self.rhs(p_tr[:, 0])
            r = self.rel(p_tr[:, 1])
            return self_tr, t*r
        return (lhs * rel) @ self.rhs.weight.t(), [(lhs, rel, rhs)]

class DisMult(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3,tao:float=0.05
    ):
        super(DisMult, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True)
            for s in sizes[:3]
        ])

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

        self.lhs = self.embeddings[0]
        self.rel = self.embeddings[1]
        self.rhs = self.embeddings[0]

    def forward(self, x,p_hr=None,p_tr=None):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        self_hr=lhs * rel
        self_tr=rhs * rel
        if p_hr is not None:
            h = self.lhs(p_hr[:, 0])
            r = self.rel(p_hr[:, 1])
            return self_hr, h*r
        if p_tr is not None:
            t = self.rhs(p_tr[:, 0])
            r = self.rel(p_tr[:, 1])
            return self_tr, t*r
        return (lhs * rel) @ self.rhs.weight.t(), [(lhs, rel, rhs)]
class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3,tao:float=0.05
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.tao=tao

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.lhs = self.embeddings[0]
        self.rel = self.embeddings[1]
        self.rhs = self.embeddings[0]

    def forward(self, x,p_hr=None,p_tr=None):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        self_hr_R = lhs[0] * rel[0] - lhs[1] * rel[1]
        self_hr_I = lhs[0] * rel[1] + lhs[1] * rel[0]
        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        if p_hr is not None:
            h = self.embeddings[0](p_hr[:, 0])
            r = self.embeddings[1](p_hr[:, 1])
            h = h[:, :self.rank], h[:, self.rank:]
            r = r[:, :self.rank], r[:, self.rank:]
            hr_R = h[0] * r[0] - h[1] * r[1]
            hr_I = h[0] * r[1] + h[1] * r[0]
            return torch.cat((self_hr_R, self_hr_I), 1), torch.cat((hr_R, hr_I), 1)
        if p_tr is not None:
            t = self.embeddings[0](p_tr[:, 0])
            r = self.embeddings[1](p_tr[:, 1])
            t = t[:, :self.rank], t[:, self.rank:]
            r = r[:, :self.rank], r[:, self.rank:]
            tr_R = t[0] * r[0] + t[1] * r[1]
            tr_I = t[0] * r[1] - t[1] * r[0]
            self_tr_R = rhs[0] * rel[0] + rhs[1] * rel[1]
            self_tr_I = rhs[0] * rel[1] - rhs[1] * rel[0]
            return torch.cat((self_tr_R, self_tr_I), 1), torch.cat((tr_R, tr_I), 1)
        ent_all=nn.functional.normalize(self.embeddings[0].weight,p=2, dim=-1)
        head = ent_all[x[:, 0]]
        tail = ent_all[x[:, 2]]


        return (
                       (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                       (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
               ), [
                   (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                    torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                    torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
               ]

