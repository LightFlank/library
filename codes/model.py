#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import ast

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE','PairRE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
        if model_name == 'PairRE' and (not double_relation_embedding):
            raise ValueError('PairRE should use --double_relation_embedding')
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'PairRE' : self.PairRE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score
    
    def PairRE(self, head, relation, tail, mode):
        re_head, re_tail = torch.chunk(relation, 2, dim=2)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        score = head * re_head - tail * re_tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score

    def cal_paire(self, tr_t , tr_r, hr_h, hr_r, head, relation, tail):
        hr_re_head, hr_re_tail = torch.chunk(hr_r, 2, dim=1)
        hr_head = F.normalize(hr_h, 2, -1)
        hr_hat = hr_head * hr_re_head
        
        tr_re_head, tr_re_tail = torch.chunk(tr_r, 2, dim=1)
        tr_tail = F.normalize(tr_t, 2, -1)
        tr_hat = tr_tail * tr_re_tail

        re_head, re_tail = torch.chunk(relation, 2, dim=1)
        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)
        hr = head * re_head
        tr = tail * re_tail
        
        return hr_hat,tr_hat,hr,tr
    
    def cal_rotate(self, tr_t, tr_r, hr_h,hr_r,head,relation,tail):
        pi = 3.14159265358979323846
        
        #hr_hat
        re_head, im_head = torch.chunk(hr_h, 2, dim=1)

        phase_relation_hr = hr_r/(self.embedding_range.item()/pi)

        re_relation_hr = torch.cos(phase_relation_hr)
        im_relation_hr = torch.sin(phase_relation_hr)

        re_score_hr = re_relation_hr * re_head - im_relation_hr * im_head
        im_score_hr = re_relation_hr * im_head + im_relation_hr * re_head

        hr_hat=torch.cat((re_score_hr,im_score_hr),-1)
        
        #tr_hat
        re_tail, im_tail = torch.chunk(tr_t, 2, dim=1)
        phase_relation_tr = tr_r/(self.embedding_range.item()/pi)
        re_relation_tr = torch.cos(phase_relation_tr)
        im_relation_tr = torch.sin(phase_relation_tr)

        re_score_tr = re_relation_tr * re_tail + im_relation_tr * im_tail
        im_score_tr = re_relation_tr * im_tail - im_relation_tr * re_tail
        tr_hat=torch.cat((re_score_tr,im_score_tr),-1)
        

        #hr and tr
        re_head, im_head = torch.chunk(head, 2, dim=1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=1)

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        tr=torch.cat((re_score,im_score),-1)
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        hr=torch.cat((re_score,im_score),-1)
        
        return hr_hat,tr_hat,hr,tr
    


    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode ,random_select, random_select_weight= next(train_iterator)
        
        positive_sample = positive_sample.cuda()

        h = torch.index_select(model.entity_embedding, dim=0, index=positive_sample[:,0])
        r = torch.index_select(model.relation_embedding, dim=0, index=positive_sample[:,1])
        t = torch.index_select(model.entity_embedding, dim=0, index=positive_sample[:,2])

        #1hop_entity
        p_h,p_h_weight = random_select[:,0],random_select_weight[:,0]
        p_t,p_t_weight = random_select[:,1],random_select_weight[:,1]

        ph = p_h.cuda()
        pt = p_t.cuda()

        p_head = torch.index_select(model.entity_embedding, dim=0, index=ph)
        p_head_w=p_h_weight.cuda()
        p_tail = torch.index_select(model.entity_embedding, dim=0, index=pt)
        p_tail_w=p_t_weight.cuda()

        squared_distances = torch.sum(torch.square(h - p_head), dim=1)
        l_r1 = torch.sum(squared_distances * p_head_w) / torch.sum(p_head_w)
        squared_distances = torch.sum(torch.square(t - p_tail), dim=1)
        l_t1 = torch.sum(squared_distances * p_tail_w) / torch.sum(p_tail_w)

        #1hop_entity+relation
        p_hr_h,p_hr_r, p_hr_weight = random_select[:,2], random_select[:,3], random_select_weight[:,2]
        p_tr_t,p_tr_r, p_tr_weight = random_select[:,4], random_select[:,5], random_select_weight[:,3]
        p_hr_h,p_hr_r = p_hr_h.cuda(), p_hr_r.cuda()
        p_tr_t,p_tr_r = p_tr_t.cuda(), p_tr_r.cuda()
        p_hr_h_embedding = torch.index_select(model.entity_embedding, dim=0, index=p_hr_h)
        p_hr_r_embedding = torch.index_select(model.relation_embedding, dim=0, index=p_hr_r)
        p_tr_t_embedding = torch.index_select(model.entity_embedding, dim=0, index=p_tr_t)
        p_tr_r_embedding = torch.index_select(model.relation_embedding, dim=0, index=p_tr_r)

        p_hr_weight=p_hr_weight.cuda()
        p_tr_weight=p_tr_weight.cuda()
        
        if args.model == 'TransE':
            #TransE计算关系loss的代码
            squared_distances = torch.sum(torch.square(h + r - p_hr_h_embedding - p_hr_r_embedding), dim=1)
            l_hr = torch.sum(squared_distances * p_hr_weight) / torch.sum(p_tr_weight)
            squared_distances = torch.sum(torch.square(t - r - p_tr_t_embedding + p_tr_r_embedding), dim=1)
            l_tr = torch.sum(squared_distances * p_tr_weight) / torch.sum(p_tr_weight)
        
        if args.model == 'RotatE':
        #RotatE计算关系loss的代码
            hr_hat,tr_hat,hr,tr = model.cal_rotate(p_tr_t_embedding,p_tr_r_embedding,p_hr_h_embedding,p_hr_r_embedding,h,r,t)
            squared_distances = torch.sum(torch.square(hr_hat-hr), dim=1)
            l_hr = torch.sum(squared_distances * p_hr_weight) / torch.sum(p_hr_weight)
            squared_distances = torch.sum(torch.square(tr_hat-tr), dim=1)
            l_tr = torch.sum(squared_distances * p_tr_weight) / torch.sum(p_tr_weight)

        if args.model == 'PairRE':
            hr_hat,tr_hat,hr,tr = model.cal_paire(p_tr_t_embedding,p_tr_r_embedding,p_hr_h_embedding,p_hr_r_embedding,h,r,t)
            squared_distances = torch.sum(torch.square(hr_hat-hr), dim=1)
            l_hr = torch.sum(squared_distances * p_hr_weight) / torch.sum(p_hr_weight)
            squared_distances = torch.sum(torch.square(tr_hat-tr), dim=1)
            l_tr = torch.sum(squared_distances * p_tr_weight) / torch.sum(p_tr_weight)
        
        '''
        entity_embedding = model.entity_embedding
        norms = torch.norm(entity_embedding, p=2, dim=1, keepdim=True)
        normalized_embeddings = entity_embedding / norms
        cosine_similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.T)
        cosine_similarity_matrix.fill_diagonal_(0)
        n = entity_embedding.size(0)
        l = cosine_similarity_matrix.sum() / (n * (n - 1))
        '''

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()
        
        loss = (positive_sample_loss + negative_sample_loss)/2 + (l_r1 + l_t1)*args.wentity + (l_hr+l_tr)*args.wentity_relation #+ l*0.01
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
