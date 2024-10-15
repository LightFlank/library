import os
import json
import argparse
import numpy as np

import torch
from torch import optim
import os
from datasets import Dataset
from models import *
from regularizers import *
from MSE_2hop_optimizers import KBCOptimizer as KBCOptimizer_high
from MSE_optimizers import KBCOptimizer
from collections import defaultdict
datasets = ['WN18RR', 'FB237','umls','kinship']

parser = argparse.ArgumentParser(
    description="Tensor Factorization for Knowledge Graph Completion"
)

parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)

parser.add_argument(
    '--model', type=str, default='CP'
)

parser.add_argument(
    '--regularizer', type=str, default='NA',
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=3, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--tao', default=0.7, type=float,
    help="temperature"
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)
parser.add_argument(
    '--we', default=0, type=float,
    help="Loss weight for E"
)
parser.add_argument(
    '--wer', default=0, type=float,
    help="Loss weight for ER"
)

parser.add_argument('-train', '--do_train', action='store_true')
parser.add_argument('-high', '--do_high', action='store_true')
parser.add_argument('-test', '--do_test', action='store_true')
parser.add_argument('-save', '--do_save', action='store_true')
parser.add_argument('-weight', '--do_ce_weight', action='store_true')
parser.add_argument('-path', '--save_path', type=str, default='../logs/')
parser.add_argument('-id', '--model_id', type=str, default='0')
parser.add_argument('-ckpt', '--checkpoint', type=str, default='')
parser.add_argument('-ckpt2', '--checkpoint2', type=str, default='')

args = parser.parse_args()

if args.do_save:
    assert args.save_path
    save_suffix = args.model + '_' + args.regularizer + '_' + args.dataset + '_' + args.model_id

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    save_path = os.path.join(args.save_path, save_suffix)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

data_path = "../data"
dataset = Dataset(data_path, args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))
if args.dataset=='FB237' and args.do_high:
    examples_train=torch.from_numpy(dataset.data['train'].astype('int64'))
else:
    examples_train=dataset.data['train']

hr2t = defaultdict(set)
tr2h = defaultdict(set)
t2hr = defaultdict(set)
h2tr = defaultdict(set)
r2ht = defaultdict(dict)
hr2rt = defaultdict(set)
tr2rh = defaultdict(set)
h2rrt = defaultdict(set)
t2rrh = defaultdict(set)

for h, r, t in examples:
    h, r, t = h.item(), r.item(), t.item()
    hr2t[(h, r)].add(t)
    tr2h[(t, r)].add(h)
    h2tr[h].add((t, r))
    t2hr[t].add((h, r))
hr2t = {k: v for k, v in hr2t.items() if v}
t2hr = {k: v for k, v in t2hr.items() if v}
h2tr = {k: v for k, v in h2tr.items() if v}
tr2h = {k: v for k, v in tr2h.items() if v}

if args.do_high:
    for h1, r1, h2 in examples_train:
        h1, r1, h2 = h1.item(), r1.item(), h2.item()
        key1 = (h1, r1)
        h2tr_cluster = h2tr.get(h2, None)
        if h2tr_cluster:
            for t, r2 in h2tr_cluster:
                hr2rt[key1].add((r2, t))
                h2rrt[h1].add((r1, r2, t))
    h2rrt = {k: v for k, v in h2rrt.items() if v}

    for h2, r2, t in examples_train:
        h2, r2, t = h2.item(), r2.item(), t.item()
        key1 = (t, r2)
        t2hr_cluster = t2hr.get(h2, None)
        if t2hr_cluster:
            for h1, r1 in t2hr_cluster:
                tr2rh[key1].add((r1, h1))
                t2rrh[t].add((r2, r1, h1))
    t2rrh = {k: v for k, v in t2rrh.items() if v}

ent2id = {}
id2ent = {}


root = os.path.join(data_path, args.dataset)


ent2idfile=os.path.join(root, "ent_id")
with open(ent2idfile, 'r') as f:
    for line in f:
        ent, id = line.strip().split()
        ent2id[ent] = int(id)
        id2ent[int(id)] = ent
entity_cnt=len(ent2id)
if args.do_ce_weight:
    ce_weight = torch.Tensor(dataset.get_weight()).cuda()
else:
    ce_weight = None

print(dataset.get_shape())

model = None
model2 = None
regularizer = None
exec('model = '+args.model+'(dataset.get_shape(), args.rank, args.init,args.tao)')
exec('model2 = '+args.model+'(dataset.get_shape(), args.rank, args.init,args.tao)')
exec('regularizer = '+args.regularizer+'(args.reg)')
regularizer = [regularizer, N3(args.reg)]

device = 'cuda'
model.to(device)
model2.to(device)
for reg in regularizer:
    reg.to(device)

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()
if args.do_high:
    if args.dataset == 'FB237':
        tr2rh.update(tr2h)
        hr2rt.update(hr2t)
        h2rrt.update(h2tr)
        t2rrh.update(t2hr)
    optimizer = KBCOptimizer_high(model, regularizer, optim_method, ent2id, args.batch_size,
                                  hr2t=hr2t, tr2h=tr2h, t2hr=t2hr, h2tr=h2tr,
                                  hr2rt=hr2rt, tr2rh=tr2rh, t2rrh=t2rrh, h2rrt=h2rrt,
                                  we=args.we, wer=args.wer)

else:
    optimizer = KBCOptimizer(model, regularizer, optim_method, ent2id, args.batch_size, hr2t=hr2t, tr2h=tr2h, t2hr=t2hr,
                             h2tr=h2tr, we=args.we, wer=args.wer)
# optimizer = KBCOptimizer(model, regularizer, optim_method, ent2id,args.batch_size)
# optimizer = KBCOptimizer(model, regularizer, optim_method, ent2id,args.batch_size,jaccard_mask_1hop=mask_1hop,jaccard_1hop_ids=jaccard_1hop_ids,hr2t=hr2t,tr2h=tr2h,t2hr=t2hr,h2tr=h2tr)


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}


cur_loss = 0

if args.checkpoint is not ''and args.checkpoint2 is not '' :
    model.load_state_dict(torch.load(os.path.join(args.checkpoint, 'checkpoint'), map_location='cuda:0'))
    model2.load_state_dict(torch.load(os.path.join(args.checkpoint2, 'checkpoint'), map_location='cuda:0'))
elif args.checkpoint is not '':
    model.load_state_dict(torch.load(os.path.join(args.checkpoint, 'checkpoint'), map_location='cuda:0'))
if args.do_test:
    valid, test, train = [
        avg_both(*dataset.eval2(model,model2, split, -1 if split != 'train' else 50000))
        for split in ['valid', 'test', 'train']
    ]
    print("\t TRAIN: ", train)
    print("\t VALID: ", valid)
    print("\t TEST: ", test)
elif args.do_train:
    with open(os.path.join(save_path, 'train.log'), 'w') as log_file:
        for e in range(args.max_epochs):
            print("Epoch: {}".format(e+1))

            cur_loss = optimizer.epoch(examples, e=e, weight=ce_weight)

            if (e + 1) % args.valid == 0:
                valid, test, train = [
                    avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                    for split in ['valid', 'test', 'train']
                ]

                print("\t TRAIN: ", train)
                print("\t VALID: ", valid)
                print("\t TEST: ", test)

                log_file.write("Epoch: {}\n".format(e+1))
                log_file.write("\t TRAIN: {}\n".format(train))
                log_file.write("\t VALID: {}\n".format(valid))
                log_file.write("\t TEST: {}\n".format(test))

                log_file.flush()

        test = avg_both(*dataset.eval(model, 'test', 50000))
        log_file.write("\t TEST: {}\n".format(test))
        print("\t TEST : ", test)

if args.do_save:
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint'))
    embeddings = model.embeddings
    len_emb = len(embeddings)
    if len_emb == 2:
        np.save(os.path.join(save_path, 'entity_embedding.npy'), embeddings[0].weight.detach().cpu().numpy())
        np.save(os.path.join(save_path, 'relation_embedding.npy'), embeddings[1].weight.detach().cpu().numpy())
    elif len_emb == 3:
        np.save(os.path.join(save_path, 'head_entity_embedding.npy'), embeddings[0].weight.detach().cpu().numpy())
        np.save(os.path.join(save_path, 'relation_embedding.npy'), embeddings[1].weight.detach().cpu().numpy())
        np.save(os.path.join(save_path, 'tail_entity_embedding.npy'), embeddings[2].weight.detach().cpu().numpy())
    else:
        print('SAVE ERROR!')

