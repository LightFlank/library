import tqdm
import torch
from torch import nn
from torch import optim
import numpy as np
from models import KBCModel
import random
from regularizers import Regularizer


class KBCOptimizer(object):
    def __init__(
            self, model: KBCModel, regularizer: list, optimizer: optim.Optimizer,ent2id:dict, batch_size: int = 256,
            verbose: bool = True,hr2t=None,tr2h=None,t2hr=None,h2tr=None,we=0,wer=0
    ):
        self.model = model
        self.regularizer = regularizer[0]
        self.regularizer2 = regularizer[1]
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.ent2id = ent2id
        self.tr2h=tr2h
        self.hr2t=hr2t
        self.t2hr=t2hr
        self.h2tr=h2tr
        self.we=we
        self.wer = wer

    def epoch(self, examples: torch.LongTensor, e=0, weight=None):
        self.model.train()
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean', weight=weight)
        p_h=[]
        p_h_weight = []
        p_hr = []
        p_hr_weight = []
        p_tr = []
        p_tr_weight = []
        p_t=[]
        p_t_weight = []
        for i in actual_examples:
            h_neibor_dict=self.tr2h[(i[2].item(),i[1].item())]
            if h_neibor_dict:
                random_h = random.choice(list(h_neibor_dict))
                if i[0]==random_h:
                    weight = 1
                else:
                    a = self.h2tr[i[0].item()]
                    b = self.h2tr[random_h]
                    intersection = a & b
                    union = len(a) + len(b) - len(intersection)
                    weight = len(intersection) / union if union > 0 else 0
                p_h.append(random_h)
                p_h_weight.append(weight)
            else:
                p_h.append(i[0].item())
                p_h_weight.append(1.0)
            t_neibor_dict=self.hr2t[(i[0].item(),i[1].item())]
            if t_neibor_dict:
                random_t = random.choice(list(t_neibor_dict))
                if i[2]==random_t:
                    weight = 1
                else:
                    a = self.t2hr[i[2].item()]
                    b = self.t2hr[random_t]
                    intersection = a & b
                    union = len(a) + len(b) - len(intersection)
                    weight = len(intersection) / union if union > 0 else 0
                p_t.append(random_t)
                p_t_weight.append(weight)
            else:
                p_t.append(i[2].item())
                p_t_weight.append(1.0)
            hr_neibor_dict = self.t2hr[i[2].item()]
            if hr_neibor_dict:
                random_hr = random.choice(list(hr_neibor_dict))
                if (i[0].item(), i[1].item())==random_hr:
                    weight = 1
                else:
                    a=self.hr2t[(i[0].item(), i[1].item())]
                    b=self.hr2t[random_hr]
                    intersection =a & b
                    union=len(a)+len(b)-len(intersection)
                    weight=len(intersection)/union if union > 0 else 0
                p_hr.append(list(random_hr))
                p_hr_weight.append(weight)
            else:
                p_hr.append(list((i[0].item(), i[1].item())))
                p_hr_weight.append(1.0)
            tr_neibor_dict = self.h2tr[i[0].item()]
            if tr_neibor_dict:
                random_tr = random.choice(list(tr_neibor_dict))
                if (i[2].item(), i[1].item())==random_tr:
                    weight = 1
                else:
                    a=self.tr2h[(i[2].item(), i[1].item())]
                    b=self.tr2h[random_tr]
                    intersection =a & b
                    union=len(a)+len(b)-len(intersection)
                    weight=len(intersection)/union if union > 0 else 0
                p_tr.append(list(random_tr))
                p_tr_weight.append(weight)
            else:
                p_tr.append(list((i[2].item(), i[1].item())))
                p_tr_weight.append(1.0)


        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()

                predictions, factors = self.model.forward(input_batch)
                lhs=self.model.lhs.weight
                rel=self.model.rel.weight
                rhs=self.model.rhs.weight
                h=lhs[input_batch[:, 0]]
                t=rhs[input_batch[:, 2]]
                r=rel[input_batch[:, 1]]
                p_head =torch.tensor(p_h[b_begin:b_begin + self.batch_size]).cuda()
                p_head = lhs[p_head]
                p_head_w =torch.tensor(p_h_weight[b_begin:b_begin + self.batch_size]).cuda()
                p_tail=torch.tensor(p_t[b_begin:b_begin + self.batch_size]).cuda()
                p_tail = rhs[p_tail]
                p_tail_w=torch.tensor(p_t_weight[b_begin:b_begin + self.batch_size]).cuda()
                p_headr = torch.tensor(p_hr[b_begin:b_begin + self.batch_size]).cuda()
                hr_emb,p_hr_emb=self.model.forward(input_batch, p_hr=p_headr)
                p_headr_w=torch.tensor(p_hr_weight[b_begin:b_begin + self.batch_size]).cuda()
                p_tailr = torch.tensor(p_tr[b_begin:b_begin + self.batch_size]).cuda()
                tr_emb,p_tr_emb=self.model.forward(input_batch, p_tr=p_tailr)
                p_tailr_w=torch.tensor(p_tr_weight[b_begin:b_begin + self.batch_size]).cuda()

                squared_distances = torch.sum(torch.square(h - p_head), dim=1)
                l3 = torch.sum(squared_distances * p_head_w) / torch.sum(p_head_w)
                squared_distances = torch.sum(torch.square(t - p_tail), dim=1)
                l4 = torch.sum(squared_distances * p_tail_w) / torch.sum(p_tail_w)
                squared_distances = torch.sum(torch.square(hr_emb - p_hr_emb), dim=1)
                l5 = torch.sum(squared_distances * p_headr_w) / torch.sum(p_headr_w)
                squared_distances = torch.sum(torch.square(tr_emb - p_tr_emb), dim=1)
                l6 = torch.sum(squared_distances * p_tailr_w) / torch.sum(p_tailr_w)


                h=nn.functional.normalize(h, dim=1)
                t=nn.functional.normalize(t, dim=1)
                hr_emb=nn.functional.normalize(hr_emb, dim=1)
                tr_emb=nn.functional.normalize(tr_emb, dim=1)
                cosine_h = torch.mm(h, h.T)
                cosine_h.fill_diagonal_(0)
                cosine_t = torch.mm(t, t.T)
                cosine_t.fill_diagonal_(0)
                cosine_hr = torch.mm(hr_emb, hr_emb.T)
                cosine_hr.fill_diagonal_(0)
                cosine_tr = torch.mm(tr_emb, tr_emb.T)
                cosine_tr.fill_diagonal_(0)
                n = h.size(0)
                l_m=(cosine_h.sum()+cosine_t.sum()+cosine_hr.sum()+cosine_tr.sum()) / (n * (n - 1)) /4

                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                l_reg = self.regularizer.forward(factors)

                l = l_fit + l_reg+(l3+l4)*self.we+(l5+l6)*self.wer+l_m*10

                self.optimizer.zero_grad()
                l.backward()

                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.1f}', reg=f'{l_reg.item():.1f}',l_h=f'{l3.item():.1f}',l_t=f'{l4.item():.1f}',l_hr=f'{l5.item():.1f}',l_tr=f'{l6.item():.1f}',l_m=f'{l_m.item():.1f}')

        return l
