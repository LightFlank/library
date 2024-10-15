#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import pickle
import random
import time

from collections import defaultdict
from torch.utils.data import Dataset

path_wn18rr='/home/wyx/KnowledgeGraphEmbedding_change_global/data/wn18rr/wn18rr_'
path_fb='/home/wyx/KnowledgeGraphEmbedding_change_global/data/FB15k-237/FB237_'
path_kinship='/home/wyx/KnowledgeGraphEmbedding_change_global/data/kinship/kinship_'
path_umls='/home/wyx/KnowledgeGraphEmbedding_change_global/data/umls/umls_'
path_=path_umls
hop = 2
#wn18rr

if hop == 1:
    #1hop
    with open(path_+'hr2t.pkl', 'rb') as f:
        hr2t = pickle.load(f)
    with open(path_+'tr2h.pkl', 'rb') as f:
        tr2h = pickle.load(f)
    with open(path_+'t2hr.pkl', 'rb') as f:
        t2hr = pickle.load(f)
    with open(path_+'h2tr.pkl', 'rb') as f:
        h2tr = pickle.load(f)

if hop == 2:
    #2hop
    with open(path_+'h2rrt.pkl', 'rb') as f:
        h2rrt = pickle.load(f)
    with open(path_+'t2rrh.pkl', 'rb') as f:
        t2rrh = pickle.load(f)
    with open(path_+'hr2tr.pkl', 'rb') as f:
        hr2tr = pickle.load(f)
    with open(path_+'tr2hr.pkl', 'rb') as f:
        tr2hr = pickle.load(f)
    with open(path_+'tr2h.pkl', 'rb') as f:
        tr2h = pickle.load(f)
    with open(path_+'hr2t.pkl', 'rb') as f:
        hr2t = pickle.load(f)

    
class TrainDataset(Dataset):

    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample
        h,r,t = head,relation,tail
        if hop == 1:
        #1hop
            tr_dict = h2tr[h]
            hr_dict = t2hr[t]
            t_dict = hr2t[(h,r)]
            h_dict = tr2h[(t,r)]

            random_h = random.choice(list(h_dict))
            random_t = random.choice(list(t_dict))
            random_hr = random.choice(list(hr_dict))
            random_tr = random.choice(list(tr_dict))

            random_select=torch.tensor([random_h,random_t,random_hr[0],random_hr[1],random_tr[0],random_tr[1]])

            h_w = TrainDataset.jaccard_1hop(h2tr[h],h2tr[random_h])
            t_w = TrainDataset.jaccard_1hop(t2hr[t],t2hr[random_t])
            hr_w = TrainDataset.jaccard_1hop(hr2t[(h,r)],hr2t[random_hr])
            tr_w = TrainDataset.jaccard_1hop(tr2h[(t,r)],tr2h[random_tr])

            random_select_weight=torch.tensor([h_w,t_w,hr_w,tr_w])
        
        
        
        if hop == 2:
            #2hop
            tr_dict = hr2tr[(h,r)]
            hr_dict = tr2hr[(t,r)]
            h_dict = tr2h[(t,r)]
            t_dict = hr2t[(h,r)]
            
            if len(tr_dict) == 0:
                tr_dict.add((t,r))
            if len(hr_dict) == 0:
                hr_dict.add((h,r))

            random_h = random.choice(list(h_dict))
            random_t = random.choice(list(t_dict))
            random_hr = random.choice(list(hr_dict))
            random_tr = random.choice(list(tr_dict))
            
            random_select=torch.tensor([random_h,random_t,random_hr[0],random_hr[1],random_tr[0],random_tr[1]])
            
            if h == random_h:
                h_w = 1
            else:
                h_w = TrainDataset.jaccard(h2rrt[h],h2rrt[random_h])
            
            if t == random_t:
                t_w = 1
            else:
                t_w = TrainDataset.jaccard(t2rrh[t],t2rrh[random_t])
            
            if (h,r) == random_hr:
                hr_w = 1
            else:
                hr_w = TrainDataset.jaccard(hr2tr[(h,r)],hr2tr[random_hr])

            if (t,r) == random_tr:
                tr_w = 1
            else:
                tr_w = TrainDataset.jaccard(tr2hr[(t,r)],tr2hr[random_tr])

            random_select_weight=torch.tensor([h_w,t_w,hr_w,tr_w])
        
        
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode, random_select, random_select_weight
    

    
    @staticmethod
    def jaccard(a,b):
        if len(a)==0 or len(b)==0:
           weight=0
        else:
            intersection_size =a&b
            union=len(a)+len(b)-len(intersection_size)
            weight=len(intersection_size)/union
        return weight
    
    def jaccard_1hop(a,b):
        if a==b:
            weight=1
        else:
            intersection_size =a&b
            union=len(a)+len(b)-len(intersection_size)
            weight=len(intersection_size)/union
        return weight




    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        random_select =torch.stack([_[4] for _ in data], dim=0)
        random_select_weight=torch.stack([_[5] for _ in data], dim=0)

        return positive_sample, negative_sample, subsample_weight, mode, random_select, random_select_weight
    

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

    
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data

