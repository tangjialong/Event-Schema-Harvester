import argparse
import pickle as pk
import os
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from nltk.corpus import stopwords
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import prettytable as pt
import random
from base.evaluater import calcACC, check_with_bcubed_lib

class AutoEncoder(nn.Module):

    def __init__(self, input_dim1, input_dim2, hidden_dims, agg, sep_decode):
        super(AutoEncoder, self).__init__()

        self.agg = agg
        self.sep_decode = sep_decode

        print("hidden_dims:", hidden_dims)
        self.encoder_layers = []
        self.encoder2_layers = []
        dims = [[input_dim1, input_dim2]] + hidden_dims
        for i in range(len(dims) - 1):
            if i == 0:
                layer = nn.Sequential(nn.Linear(dims[i][0], dims[i+1]), nn.ReLU())
                layer2 = nn.Sequential(nn.Linear(dims[i][1], dims[i+1]), nn.ReLU())
            elif i != 0 and i < len(dims) - 2:
                layer = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
                layer2 = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
            else:
                layer = nn.Linear(dims[i], dims[i+1])
                layer2 = nn.Linear(dims[i], dims[i+1])
            self.encoder_layers.append(layer)
            self.encoder2_layers.append(layer2)
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.encoder2 = nn.Sequential(*self.encoder2_layers)

        self.decoder_layers = []
        self.decoder2_layers = []
        hidden_dims.reverse()
        dims = hidden_dims + [[input_dim1, input_dim2]]
        if self.agg == "concat" and not self.sep_decode:
            dims[0] = 2 * dims[0]
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layer = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
                layer2 = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
            else:
                layer = nn.Linear(dims[i], dims[i+1][0])
                layer2 = nn.Linear(dims[i], dims[i+1][1])
            self.decoder_layers.append(layer)
            self.decoder2_layers.append(layer2)
        self.decoder = nn.Sequential(*self.decoder_layers)
        self.decoder2 = nn.Sequential(*self.decoder2_layers)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder2(x2)

        if self.agg == "max":
            z = torch.max(z1, z2)
        elif self.agg == "multi":
            z = z1 * z2
        elif self.agg == "sum":
            z = z1 + z2
        elif self.agg == "concat":
            z = torch.cat([z1, z2], dim=1)

        if self.sep_decode:
            x_bar1 = self.decoder(z1)
            x_bar1 = F.normalize(x_bar1, dim=-1)
            x_bar2 = self.decoder2(z2)
            x_bar2 = F.normalize(x_bar2, dim=-1)
        else:
            x_bar1 = self.decoder(z)
            x_bar1 = F.normalize(x_bar1, dim=-1)
            x_bar2 = self.decoder2(z)
            x_bar2 = F.normalize(x_bar2, dim=-1)

        return x_bar1, x_bar2, z

class TopicCluster(nn.Module):

    def __init__(self, dataset_path, device, temperature, distribution, agg_method, sep_decode, input_dim1, input_dim2, hidden_dims, batch_size, lr, n_clusters):
        super(TopicCluster, self).__init__()
        self.alpha = 1.0
        self.dataset_path = dataset_path
        self.device = device
        self.temperature = temperature
        self.distribution = distribution
        self.agg_method = agg_method
        self.sep_decode = (sep_decode == 1)
        self.batch_size = batch_size
        self.lr = lr

        input_dim1 = input_dim1
        input_dim2 = input_dim2
        hidden_dims = eval(hidden_dims)
        self.model = AutoEncoder(input_dim1, input_dim2, hidden_dims, self.agg_method, self.sep_decode)
        if self.agg_method == "concat":
            self.topic_emb = Parameter(torch.Tensor(n_clusters, 2*hidden_dims[-1]))
        else:
            self.topic_emb = Parameter(torch.Tensor(n_clusters, hidden_dims[-1]))
        torch.nn.init.xavier_normal_(self.topic_emb.data)

    def pretrain(self, input_data, load_pretrain, pretrain_epoch=1000):
        pretrained_path = os.path.join(self.dataset_path, "pretrained_etypeclus_result.pt")
        if os.path.exists(pretrained_path) and load_pretrain:
            # load pretrain weights
            print(f"loading pretrained model from {pretrained_path}")
            self.model.load_state_dict(torch.load(pretrained_path))
        else:
            train_loader = DataLoader(input_data, batch_size=self.batch_size, shuffle=True)
            optimizer = Adam(self.model.parameters(), lr=self.lr)
            for epoch in range(pretrain_epoch):
                total_loss = 0
                for batch_idx, (x1, x2, _, weight) in enumerate(train_loader):
                    x1 = x1.to(self.device)
                    x2 = x2.to(self.device)
                    weight = weight.to(self.device)
                    optimizer.zero_grad()
                    x_bar1, x_bar2, z = self.model(x1, x2)
                    loss = cosine_dist(x_bar1, x1) + cosine_dist(x_bar2, x2) #, weight)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                print(f"epoch {epoch}: loss = {total_loss / (batch_idx+1):.4f}")
                if total_loss / (batch_idx+1) < 0.25:
                    break
            torch.save(self.model.state_dict(), pretrained_path)
            print(f"model saved to {pretrained_path}")

    def cluster_assign(self, z):
        if self.distribution == 'student':
            p = 1.0 / (1.0 + torch.sum(
                torch.pow(z.unsqueeze(1) - self.topic_emb, 2), 2) / self.alpha)
            p = p.pow((self.alpha + 1.0) / 2.0)
            p = (p.t() / torch.sum(p, 1)).t()
        else:
            self.topic_emb.data = F.normalize(self.topic_emb.data, dim=-1)
            z = F.normalize(z, dim=-1)
            sim = torch.matmul(z, self.topic_emb.t()) / self.temperature
            p = F.softmax(sim, dim=-1)
        return p
    
    def forward(self, x1, x2):
        x_bar1, x_bar2, z = self.model(x1, x2)
        p = self.cluster_assign(z)
        return x_bar1, x_bar2, z, p

    def target_distribution(self, x1, x2, freq, method='all', top_num=0):
        _, _, z = self.model(x1, x2)
        p = self.cluster_assign(z).detach()
        if method == 'all':
            q = p**2 / (p * freq.unsqueeze(-1)).sum(dim=0)
            q = (q.t() / q.sum(dim=1)).t()
        elif method == 'top':
            assert top_num > 0
            q = p.clone()
            sim = torch.matmul(self.topic_emb, z.t())
            _, selected_idx = sim.topk(k=top_num, dim=-1)
            for i, topic_idx in enumerate(selected_idx):
                q[topic_idx] = 0
                q[topic_idx, i] = 1
        return p, q

def cosine_dist(x_bar, x, weight=None):
    if weight is None:
        weight = torch.ones(x.size(0), device=x.device)
    cos_sim = (x_bar * x).sum(-1)
    cos_dist = 1 - cos_sim
    cos_dist = (cos_dist * weight).sum() / weight.sum()
    return cos_dist

def etyeclus(data_dict, id_map, k, output_dir, times=10, 
            use_freq=False, 
            distribution='softmax', 
            agg_method='concat', 
            sep_decode=0, 
            hidden_dims='[500, 500, 1000, 100]', 
            pretrain_epoch=1000, 
            total_epoch=100,
            update_interval=100,
            load_pretrain=False,
            batch_size=64,
            lr=0.001,
            tol=0.05,
            gamma=0.02,
            temperature=0.1, 
            sort_method='discriminative'):
    '''
        distribution, choices=['softmax', 'student']
        agg_method, choices=["max", "sum", "multi", "concat"]
        sep_decode, choices=[0, 1]
        sort_method, choices=['generative', 'discriminative']
    '''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    emb_dict = data_dict
    n_clusters = k

    inv_vocab = {k: " ".join(v) for k, v in emb_dict["inv_vocab"].items()}
    vocab = {" ".join(k):v for k, v in emb_dict["vocab"].items()}
    print (f"Vocab size: {len(vocab)}")
    embs = F.normalize(torch.tensor(emb_dict["vs_emb"]), dim=-1)
    embs2 = F.normalize(torch.tensor(emb_dict["oh_emb"]), dim=-1)
    freq = np.array(emb_dict["tuple_freq"])
    if not use_freq:
        freq = np.ones_like(freq)
    input_data = TensorDataset(embs, embs2, torch.arange(embs.size(0)), torch.tensor(freq))
    cuda = torch.cuda.is_available()
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")
    input_dim1 = emb_dict['vs_emb'].shape[1]
    input_dim2 = emb_dict['oh_emb'].shape[1]

    golden = [id_map.index(v) for k, v in data_dict['event_type'].items()]
    random.seed(1234)
    np.random.seed(1234)

    all_ARI = []
    all_NMI = []
    all_ACC = []
    all_Bcubed_F1 = []
    all_clusters = []
    for time in range(times):
        time_output_dir = os.path.join(output_dir, str(time))
        if not os.path.exists(time_output_dir):
            os.mkdir(time_output_dir)
        topic_cluster = TopicCluster(time_output_dir, device, temperature, distribution, agg_method, sep_decode, input_dim1, input_dim2, hidden_dims, batch_size, lr, n_clusters).to(device)
        topic_cluster.pretrain(input_data, load_pretrain, pretrain_epoch)
        train_loader = DataLoader(input_data, batch_size=batch_size, shuffle=True)
        optimizer = Adam(topic_cluster.parameters(), lr=lr)

        # topic embedding initialization
        embs = embs.to(device)
        embs2 = embs2.to(device)
        x_bar1, x_bar2, z = topic_cluster.model(embs, embs2)
        z = F.normalize(z, dim=-1)

        print (f"Running K-Means for initialization")
        kmeans = KMeans(n_clusters=n_clusters, n_init=5)
        if use_freq:
            y_pred = kmeans.fit_predict(z.data.cpu().numpy(), sample_weight=freq)
        else:
            y_pred = kmeans.fit_predict(z.data.cpu().numpy())
        print (f"Finish K-Means")
        freq = torch.tensor(freq).to(device)
        y_pred_last = y_pred
        topic_cluster.topic_emb.data = torch.tensor(kmeans.cluster_centers_).to(device)

        topic_cluster.train()
        i = 0
        total_loss = 0
        total_kl_loss = 0
        total_rec_loss = 0
        for epoch in range(total_epoch):
            flag = False
            for x1, x2, idx, weight in train_loader:
                if i % update_interval == 0:
                    p, q = topic_cluster.target_distribution(embs, embs2, freq.clone().fill_(1), method='all', top_num=epoch+1)

                    y_pred = p.cpu().numpy().argmax(1)
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = y_pred

                    ARI = adjusted_rand_score(golden, y_pred_last)
                    NMI = normalized_mutual_info_score(golden, y_pred_last)
                    ACC = calcACC(golden, y_pred_last)
                    Bcubed_F1 = check_with_bcubed_lib(golden, y_pred_last)

                    _, _, z, p = topic_cluster(embs, embs2)
                    z = F.normalize(z, dim=-1)
                    topic_cluster.topic_emb.data = F.normalize(topic_cluster.topic_emb.data, dim=-1)
                    if not os.path.exists(os.path.join(time_output_dir, "clusters_etypeclus_result")):
                        os.makedirs(os.path.join(time_output_dir, "clusters_etypeclus_result"))
                    f = open(os.path.join(time_output_dir, f"clusters_etypeclus_result/{i}.json"), 'w')
                    pred_cluster = p.argmax(-1)

                    result_strings = []
                    for j in range(n_clusters):
                        if sort_method == 'discriminative':
                            word_idx = torch.arange(embs.size(0))[pred_cluster == j]
                            sorted_idx = torch.argsort(p[pred_cluster == j][:, j], descending=True)
                            word_idx = word_idx[sorted_idx]
                        else:
                            sim = torch.matmul(topic_cluster.topic_emb[j], z.t())
                            _, word_idx = sim.topk(k=30, dim=-1)
                        word_cluster = []
                        freq_sum = 0
                        for idx in word_idx:
                            freq_sum += freq[idx].item()
                            if inv_vocab[idx.item()] not in word_cluster:
                                word_cluster.append(inv_vocab[idx.item()])
                        result_strings.append((freq_sum, f"Topic {j} ({freq_sum}): " + ', '.join(word_cluster)+'\n'))
                    for result_string in result_strings:
                        f.write(result_string[1])
                    
                    if i > 0:
                        print(f"Loss: {loss/i:.4f}; KL loss: {kl_loss/i:.4f}; Reconstruction loss: {reconstr_loss/i:.4f}; Delta label: {delta_label:.4f}")
                    if i > 0 and delta_label < tol:
                        print(f'delta_label {delta_label:.4f} < tol ({tol})')
                        print('Reached tolerance threshold. Stopping training.')
                        all_ARI.append(ARI)
                        all_NMI.append(NMI)
                        all_ACC.append(ACC)
                        all_Bcubed_F1.append(Bcubed_F1)
                        all_clusters.append(n_clusters)
                        flag = True
                        break

                i += 1
                x1 = x1.to(device)
                x2 = x2.to(device)
                idx = idx.to(device)
                weight = weight.to(device)

                x_bar1, x_bar2, _, p = topic_cluster(x1, x2)
                reconstr_loss = cosine_dist(x_bar1, x1) + cosine_dist(x_bar2, x2) #, weight)
                kl_loss = F.kl_div(p.log(), q[idx], reduction='none').sum(-1)
                kl_loss = (kl_loss * weight).sum() / weight.sum()
                loss = gamma * kl_loss + reconstr_loss
                total_kl_loss += kl_loss.item()
                total_rec_loss += reconstr_loss.item()
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if flag: break

    return np.array(all_ARI).mean(), np.array(all_NMI).mean(), np.array(all_ACC).mean(), np.array(all_Bcubed_F1).mean(), np.array(all_clusters).mean(), \
        np.array(all_ARI).std(), np.array(all_NMI).std(), np.array(all_ACC).std(), np.array(all_Bcubed_F1).std(), np.array(all_clusters).std()