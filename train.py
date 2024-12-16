import argparse
import datetime
import os
import random

import torch.nn.functional as F
import yaml
from tqdm import tqdm
import numpy as np
import torch

from TriCL.loader import DatasetLoader
from TriCL.models import HyperEncoder, TriCL
from TriCL.utils import drop_features, drop_incidence, valid_node_edge_mask, hyperedge_index_masking
from TriCL.utils import drop_incidence_based_on_similarity_multiround
from TriCL.evaluation import linear_evaluation

from datetime import datetime

p_test_flag = 0  # 1是测0.1-1.0 rate, 2 是测20个seed, 其他为正常跑 
drop_incidence_rate_flag =0  # 1是测0.1-1.0 rate，其他为正常跑
drop_feature_rate_flag = 0     
custom_text = " new simi conv"

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train(model_type, epoch, num_negs, seed):
    features, hyperedge_index = data.features, data.hyperedge_index
    num_nodes, num_edges = data.num_nodes, data.num_edges

    model.train()
    optimizer.zero_grad(set_to_none=True)

    hyperedge_index1 = drop_incidence_based_on_similarity_multiround(features,params['drop_incidence_rate'],hyperedge_index)
    hyperedge_index2 = drop_incidence_based_on_similarity_multiround(features,params['drop_incidence_rate'],hyperedge_index)

    x1 = drop_features(features, params['drop_feature_rate'])
    x2 = drop_features(features, params['drop_feature_rate'])

    node_mask1, edge_mask1 = valid_node_edge_mask(hyperedge_index1, num_nodes, num_edges)
    node_mask2, edge_mask2 = valid_node_edge_mask(hyperedge_index2, num_nodes, num_edges)
    node_mask = node_mask1 & node_mask2
    edge_mask = edge_mask1 & edge_mask2

    # Encoder
    n1, e1 = model(x1, hyperedge_index1, num_nodes, num_edges)
    n2, e2 = model(x2, hyperedge_index2, num_nodes, num_edges)

    # Projection Head
    n1, n2 = model.node_projection(n1), model.node_projection(n2)
    e1, e2 = model.edge_projection(e1), model.edge_projection(e2)

    # 计算训练损失
    loss_n = model.node_level_loss(n1, n2, params['tau_n'], batch_size=params['batch_size_1'], num_negs=num_negs)
    if model_type in ['tricl_ng', 'tricl']:
        loss_g = model.group_level_loss(e1[edge_mask], e2[edge_mask], params['tau_g'], batch_size=params['batch_size_1'], num_negs=num_negs)
    else:
        loss_g = 0

    if model_type in ['tricl']:
        masked_index1 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, None, edge_mask1)
        masked_index2 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, None, edge_mask2)
        loss_m1 = model.membership_level_loss(n1, e2[edge_mask2], masked_index2, params['tau_m'], batch_size=params['batch_size_2'])
        loss_m2 = model.membership_level_loss(n2, e1[edge_mask1], masked_index1, params['tau_m'], batch_size=params['batch_size_2'])
        loss_m = (loss_m1 + loss_m2) * 0.5
    else:
        loss_m = 0

    loss = loss_n + params['w_g'] * loss_g + params['w_m'] * loss_m 
    loss.backward()
    optimizer.step()
    return loss.item()


def node_classification_eval(num_splits=20):
    model.eval()
    n, _ = model(data.features, data.hyperedge_index)

    if data.name == 'pubmed':
        lr = 0.005
        max_epoch = 300
    elif data.name == 'cora' or data.name == 'citeseer':
        lr = 0.005
        max_epoch = 100
    elif data.name == 'Mushroom':
        lr = 0.01
        max_epoch = 200
    else:
        lr = 0.01
        max_epoch = 100

    accs = []
    for i in range(num_splits):
        masks = data.generate_random_split(seed=i)
        accs.append(linear_evaluation(n, data.labels, masks, lr=lr, max_epoch=max_epoch))
    return accs 


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TriCL unsupervised learning.')
    parser.add_argument('--dataset', type=str, default='cora', 
        choices=['cora', 'citeseer', 'pubmed', 'cora_coauthor', 'dblp_coauthor', 
                 'zoo', '20newsW100', 'Mushroom', 'NTU2012', 'ModelNet40'])
    parser.add_argument('--model_type', type=str, default='tricl', choices=['tricl_n', 'tricl_ng', 'tricl'])
    if p_test_flag==1:
        parser.add_argument('--num_seeds', type=int, default=10)
    elif p_test_flag==2:
        parser.add_argument('--num_seeds', type=int, default=20)       
    else:
        parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()

    params = yaml.safe_load(open('Semester_2\\TriCL_hater\\config.yaml'))[args.dataset]
    print(params)

    data = DatasetLoader().load(args.dataset).to(args.device)

    accs = []
    file_path = os.path.join('Semester_2', 'TriCL_hater', 'Log', f"{args.dataset}.txt")
    mode = 'a' if os.path.isfile(file_path) else 'w'

    with open(file_path, mode) as log_f:   
        log_f.write(f"\n")
        log_f.write(f"dataset: {args.dataset} \n")
        log_f.write(f"params:{params}\n")
        log_f.write(f'notice:{custom_text}\n')

        if p_test_flag == 2:
             log_f.write(f"test best evalseed\n")
        elif drop_incidence_rate_flag == 1 and p_test_flag == 1:
             log_f.write(f"test best drop_incidence_rate\n")
        elif drop_feature_rate_flag == 1 and p_test_flag == 1:
             log_f.write(f"test best drop_feature_rate\n")
        else:
             log_f.write(f"normal run\n")       

        for seed in range(args.num_seeds):
            if p_test_flag==1:
                fix_seed(1)
            elif p_test_flag==2:
                fix_seed(4)                
            else:
                fix_seed(seed)
            encoder = HyperEncoder(data.features.shape[1], params['hid_dim'], params['hid_dim'], params['num_layers']).to(args.device)
            model = TriCL(encoder, params['proj_dim']).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

            for epoch in tqdm(range(1, params['epochs'] + 1)):
                loss = train(args.model_type, epoch, num_negs=None, seed=seed)              
            acc = node_classification_eval(20)

            accs.append(acc)
            acc_mean, acc_std = np.mean(acc, axis=0), np.std(acc, axis=0)
            print(f'seed: {seed}, train_acc: {acc_mean[0]:.2f}+-{acc_std[0]:.2f}, '
                f'valid_acc: {acc_mean[1]:.2f}+-{acc_std[1]:.2f}, test_acc: {acc_mean[2]:.2f}+-{acc_std[2]:.2f}')
            
            if p_test_flag==1:
                log_f.write(f'rate: {seed/10}, train_acc: {acc_mean[0]:.2f}+-{acc_std[0]:.2f}, '
                    f'valid_acc: {acc_mean[1]:.2f}+-{acc_std[1]:.2f}, test_acc: {acc_mean[2]:.2f}+-{acc_std[2]:.2f}\n')
            elif p_test_flag==2:
                log_f.write(f'valseed: {seed}, train_acc: {acc_mean[0]:.2f}+-{acc_std[0]:.2f}, '
                    f'valid_acc: {acc_mean[1]:.2f}+-{acc_std[1]:.2f}, test_acc: {acc_mean[2]:.2f}+-{acc_std[2]:.2f}\n')
            else:
                log_f.write(f'seed: {seed}, train_acc: {acc_mean[0]:.2f}+-{acc_std[0]:.2f}, '
                    f'valid_acc: {acc_mean[1]:.2f}+-{acc_std[1]:.2f}, test_acc: {acc_mean[2]:.2f}+-{acc_std[2]:.2f}\n')
 
        accs = np.array(accs).reshape(-1, 3)
        accs_mean = list(np.mean(accs, axis=0))
        accs_std = list(np.std(accs, axis=0))
        print(f'[Final] dataset: {args.dataset}, test_acc: {accs_mean[2]:.2f}+-{accs_std[2]:.2f}')


        # Write final test accuracy
        log_f.write(f'mean test_acc: {accs_mean[2]:.2f}+-{accs_std[2]:.2f}\n')

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_f.write(f"datetim:{current_datetime} \n")

        