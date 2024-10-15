
# RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space
**Introduction**


**Implemented features**

Models:
 - [x] RotatE
 - [x] TransE
 - [x] pairRE


Evaluation Metrics:

 - [x] MRR, MR, HITS@1, HITS@3, HITS@10 (filtered)

Loss Function:

 - [x] Uniform Negative Sampling
 - [x] Self-Adversarial Negative Sampling

**Usage**

Knowledge Graph Data:
 - *entities.dict*: a dictionary map entities to unique ids
 - *relations.dict*: a dictionary map relations to unique ids
 - *train.txt*: the KGE model is trained to fit this data set
 - *valid.txt*: create a blank file if no validation data is available
 - *test.txt*: the KGE model is evaluated on this data set

**Train**

 CUDA_VISIBLE_DEVICES=1 python codes/run.py --do_train --do_valid --do_test --cuda --data_path data/umls --model RotatE -n 512 -b 64 -d 1000 -g 2.0 -a 1.0 -adv -lr 0.00001 --max_steps 20000 -we 0.7 -wer 0.2 -save models/RotatE_umls_1hop0702 --test_batch_size 8 -de

For example, this command train a RotatE model on umls dataset with GPU 0 and we is 0.7 and wer is 0.2 on First Order.
```
First you should change the parameters on 

 CUDA_VISIBLE_DEVICES=1 python codes/run.py --do_train --do_valid --do_test --cuda --data_path data/umls --model RotatE -n 512 -b 64 -d 1000 -g 2.0 -a 1.0 -adv -lr 0.00001 --max_steps 20000 -we 0.7 -wer 0.2 -save models/RotatE_umls_1hop0702 --test_batch_size 8 -de
```
   Check argparse configuration at codes/run.py for more arguments and more details.

**Test**

    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE

**Reproducing the best results**

In order to produce the best results, we parameters are generally set to 0.6,0.8 or 0.4 and wer is generally set to 0.1

Refer to the previous parameters:
bash run.sh train RotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16 -de
bash run.sh train RotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8 -de
bash run.sh train PairRE FB15k-237 0 0 1024 256 1500 6.0 1.0 0.00005 100000 16 -dr

bash run.sh train pRotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16
bash run.sh train pRotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8


# Best Configuration for TransE
bash run.sh train TransE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16
bash run.sh train TransE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8

# Best Configuration for ComplEx
bash run.sh train ComplEx FB15k-237 0 0 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
bash run.sh train ComplEx wn18rr 0 0 512 1024 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005

# Best Configuration for DistMult
bash run.sh train DistMult FB15k-237 0 0 1024 256 2000 200.0 1.0 0.001 100000 16 -r 0.00001
bash run.sh train DistMult wn18rr 0 0 512 1024 1000 200.0 1.0 0.002 80000 8 -r 0.000005


there are some examples:
remember to change folder address in dataloader.py

CUDA_VISIBLE_DEVICES=0 python codes/run.py --do_train  --cuda  --do_valid  --do_test  --data_path data/umls --model PairRE  -n 256 -b 1024 -d 1500  -g 2.0 -a 1.0 -adv  -lr 0.00003 --max_steps 20000 -we 1.0 -wer 0.1 -dr -save models/PairRE_yuan --test_batch_size 8

CUDA_VISIBLE_DEVICES=1 python codes/run.py --do_train --do_valid --do_test --cuda --data_path data/kinship --model TransE -n 512 -b 64 -d 1000 -g 1.25 -a 1.0 -adv -lr 0.00002 --max_steps 20000 -we 0.6 -wer 0.1 -save models/TransE_kinship_2wstep_1 --test_batch_size 8


CUDA_VISIBLE_DEVICES=0 nohup python -u codes/run.py --do_train --do_valid --do_test --cuda --data_path data/kinship --model TransE -n 512 -b 128 -d 1000 -g 1.25 -a 1.0 -adv -lr 0.000
01 --max_steps 50000 -we 0.6 -wer 0.1 -save models/TransE_kinship --test_batch_size 8 