import numpy as np
from numpy.core.fromnumeric import shape
import scipy.sparse as sp
import torch
import torch.nn as nn
from aug import *
from model import *
from utils import *
from aug import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler

import random
import os
import dgl

import argparse
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set argument
parser = argparse.ArgumentParser(description='Sub')
parser.add_argument('--dataset', type=str,default='citeseer')  # 'BlogCatalog'  'Flickr'   'cora'  'citeseer'  'pubmed'
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=300)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.5)

args = parser.parse_args()

if args.lr is None:
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'Flickr']:
        args.lr = 1e-3
    elif args.dataset == 'BlogCatalog':
        args.lr = 3e-3

if args.num_epoch is None:
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        args.num_epoch = 100
    elif args.dataset in ['BlogCatalog', 'Flickr']:
        args.num_epoch = 400

batch_size = args.batch_size
subgraph_size = args.subgraph_size

print('Dataset: ', args.dataset)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load and preprocess data
adj, features, labels, idx_train, idx_val, \
idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)
# if the folder don't have diffusion data, the code for generate: 
#diff=gdc(adj,alpha=0.01,eps=0.0001)
#np.save('diff_A',diff)  A can be changed to 'BlogCatalog' 'cite' 'Flickr'...
# if the folder has diffusion data:
diff = np.load('./diff_citeseer.npy' ,allow_pickle=True)

b_adj = sp.csr_matrix(diff)
b_adj = (b_adj + sp.eye(b_adj.shape[0])).todense()
dgl_graph = adj_to_dgl_graph(adj)
raw_feature=features.todense()
features, _ = preprocess_features(features)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
adj = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
raw_feature = torch.FloatTensor(raw_feature[np.newaxis])

adj = torch.FloatTensor(adj[np.newaxis])
b_adj = torch.FloatTensor(b_adj[np.newaxis])

# Initialize model and optimiser
model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout, args.dropout)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if torch.cuda.is_available():
    print('Using CUDA')
    model.to(device)
    features = features.to(device)
    raw_feature = raw_feature.to(device)
    adj = adj.to(device)
    b_adj = b_adj.to(device)

if torch.cuda.is_available():
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
else:
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0
batch_num = nb_nodes // batch_size + 1

added_adj_zero_row = torch.zeros((nb_nodes, 1, subgraph_size))
added_adj_zero_col = torch.zeros((nb_nodes, subgraph_size + 1, 1))
added_adj_zero_col[:, -1, :] = 1.
added_feat_zero_row = torch.zeros((nb_nodes, 1, ft_size))
if torch.cuda.is_available():
    added_adj_zero_row = added_adj_zero_row.to(device)
    added_adj_zero_col = added_adj_zero_col.to(device)
    added_feat_zero_row = added_feat_zero_row.to(device)
mse_loss = nn.MSELoss(reduction='mean')
# # Train model
with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Training')
    for epoch in range(args.num_epoch):

        loss_full_batch = torch.zeros((nb_nodes, 1))
        if torch.cuda.is_available():
            loss_full_batch = loss_full_batch.to(device)

        model.train()

        all_idx = list(range(nb_nodes))

        random.shuffle(all_idx)
        total_loss = 0.
        subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
        p = 0
        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num))

            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)

            lbl = torch.unsqueeze(
                torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1)

            ba = []
            bf = []
            br = []
            raw=[]

            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

            if torch.cuda.is_available():
                lbl = lbl.to(device)
                added_adj_zero_row = added_adj_zero_row.to(device)
                added_adj_zero_col = added_adj_zero_col.to(device)
                added_feat_zero_row = added_feat_zero_row.to(device)

            for i in idx:
                cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_adj_r = b_adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_feat = features[:, subgraphs[i], :]
                raw_f=raw_feature[:, subgraphs[i], :]
                ba.append(cur_adj)
                bf.append(cur_feat)
                raw.append(raw_f)
                br.append(cur_adj_r)


            ba = torch.cat(ba)
            br = torch.cat(br)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)

            br = torch.cat((br, added_adj_zero_row), dim=1)
            br = torch.cat((br, added_adj_zero_col), dim=2)

            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

            raw = torch.cat(raw)
            raw = torch.cat((raw[:, :-1, :], added_feat_zero_row, raw[:, -1:, :]), dim=1)

            now1, logits = model(bf, ba,raw)
            now2, logits2 = model(bf, br,raw)
            batch = now1.shape[0]
            loss_re=0.5 * (mse_loss(now1, raw[:, -1, :]) + mse_loss(now2, raw[:, -1, :]))
            loss_all2 = b_xent(logits2, lbl)
            loss_all1 = b_xent(logits, lbl)
            loss_bce = (loss_all1 + loss_all2) /2
            h_1 = F.normalize(logits[:batch, :], dim=1, p=2)
            h_2 = F.normalize(logits2[:batch, :], dim=1, p=2)
            coloss2 = 2 - 2 * (h_1 * h_2).sum(dim=-1).mean()
            loss = torch.mean(loss_bce) + coloss2+0.6*loss_re

            loss.backward()   
            optimiser.step()

            loss = loss.detach().cpu().numpy()

            total_loss += loss
            p = p + 1

        mean_loss = total_loss

        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_model.pkl')  # multi_round_ano_score_p[round, idx] = ano_score_p

        else:
            cnt_wait += 1

        pbar.update(1)

# # # # # Test model
print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_model.pkl'))

multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score_p = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score_n = np.zeros((args.auc_test_rounds, nb_nodes))
kk = 0

with tqdm(total=args.auc_test_rounds) as pbar_test:
    pbar_test.set_description('Testing')
    for round in range(args.auc_test_rounds):

        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)

        subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num - 1))

            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)

            ba = []
            bf = []
            br = []
            raw=[]
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

            if torch.cuda.is_available():
                added_adj_zero_row = added_adj_zero_row.to(device)
                added_adj_zero_col = added_adj_zero_col.to(device)
                added_feat_zero_row = added_feat_zero_row.to(device)

            for i in idx:
                cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_adj2 = b_adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_feat = features[:, subgraphs[i], :]
                raw_f = raw_feature[:, subgraphs[i], :]
                ba.append(cur_adj)
                br.append(cur_adj2)
                bf.append(cur_feat)
                raw.append(raw_f)

            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            br = torch.cat(br)
            br = torch.cat((br, added_adj_zero_row), dim=1)
            br = torch.cat((br, added_adj_zero_col), dim=2)
            
            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
            raw = torch.cat(raw)
            raw = torch.cat((raw[:, :-1, :], added_feat_zero_row, raw[:, -1:, :]), dim=1)

            with torch.no_grad():
                now1, logits = model(bf, ba,raw)
                now2, logits2 = model(bf, br,raw)

                logits = torch.squeeze(logits)
                logits = torch.sigmoid(logits)

                logits2 = torch.squeeze(logits2)
                logits2 = torch.sigmoid(logits2)
            pdist = nn.PairwiseDistance(p=2)
            

            scaler1 = MinMaxScaler()
            scaler2 = MinMaxScaler()
            ano_score1 = - (logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()
            ano_score2 = - (logits2[:cur_batch_size] - logits2[cur_batch_size:]).cpu().numpy()
            score1=(pdist(now1, raw[:, -1, :])+pdist(now2, raw[:, -1, :]))/2
            
            score2=(ano_score1+ano_score2)/2
            score1=score1.cpu().numpy()
            ano_score_co = scaler1.fit_transform(score2.reshape(-1, 1)).reshape(-1)
            score_re = scaler2.fit_transform(score1.reshape(-1, 1)).reshape(-1)
            ano_scores = ano_score_co+0.6*score_re
            multi_round_ano_score[round, idx] = ano_scores

        pbar_test.update(1)

ano_score_final = np.mean(multi_round_ano_score, axis=0)
auc = roc_auc_score(ano_label, ano_score_final)
print('AUC:{:.4f}'.format(auc))

