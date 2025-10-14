import numpy as np
import torch
import itertools as its
from sklearn.metrics import roc_auc_score, average_precision_score
torch.set_default_dtype(torch.float32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def merge_set(groups, l, r):
    for idx, i in enumerate(groups):
        if l in i:
            left_idx = idx
        if r in i:
            right_idx = idx
    if left_idx==right_idx:
        return groups
    # new = np.concatenate([groups[left_idx], groups[right_idx]], axis=1)
    l_all, r_all= groups[left_idx], groups[right_idx]
    new = l_all+r_all
    new.sort()
    groups.remove(l_all)
    groups.remove(r_all)
    return sorted([new] + groups)

class GDAD(object):
    def __init__(self):
        self.p = 0.8    # default
        self.tau = 0.5  # default
        self.mi = None
        self.attr_P = None

    def grid_search(self, train_X, train_y, paras, nominals=None):
        self.n, self.m = train_X.shape
        self.nominals = nominals
        self.data = torch.from_numpy(train_X.T).float().unsqueeze(-1).to(device)
        self.mutual_information(train_X)
        self.attribute_partition()

        records = dict()
        for p in paras:
            out_scores = self.predict_score(p=p)
            auc = roc_auc_score(train_y, out_scores)
            pr = average_precision_score(y_true=train_y, y_score=out_scores, pos_label=1)
            records[p] = auc + pr
            print('\tp={}\tAUC={:.4f}\tPR={:.4f}\tAUC+PR={:.4f}'.format(p, auc, pr, auc+pr))
        self.p = max(records.keys(), key=lambda k:records[k])


    def mutual_information(self, data):
        discretized_data = data.T.copy()
        entropy = np.zeros(self.m, dtype=np.float32)
        for idx in range(self.m):
            if not self.nominals[idx]:
                # 使用 numpy.histogram_bin_edges 直接获取最优的 bin 边界，进而将数值型数据离散化
                temp = discretized_data[idx]
                bin_edges = np.histogram_bin_edges(temp, bins='auto')
                discretized_data[idx] = np.digitize(temp, bins=bin_edges[:-1],  # 忽略最后一个边界（np.digitize的特殊要求）
                                                    right=True)
            # 对每个维度计算信息熵
            _, counts = np.unique(discretized_data[idx], return_counts=True)
            probabilities = counts / counts.sum()
            entropy[idx] = -(probabilities * np.log2(probabilities)).sum()
        # print('Entropy of attribues:',entropy)

        idx = 0
        self.mi = np.zeros(self.m*(self.m-1)//2, dtype=np.float32)
        for x,y in its.combinations(np.arange(self.m), 2):
            # 计算联合概率分布
            xy = np.stack([discretized_data[x], discretized_data[y]], axis=1)
            _, xy_counts = np.unique(xy, axis=0, return_counts=True)
            prob_xy = xy_counts / xy_counts.sum()

            # 计算联合熵
            entropy_xy = -(prob_xy * np.log2(prob_xy)).sum()

            # 计算互信息
            self.mi[idx] = (entropy[x] + entropy[y] - entropy_xy) / np.sqrt(entropy[x] * entropy[y])
            # print(f"entropy({x}): {entropy[x]}, entropy({y}): {entropy[y]}, entropy({x},{y}):{entropy_xy}, mi:{self.mi[idx]}")
            idx += 1


    def attribute_partition(self):
        subgroups = np.arange(self.m).reshape(-1, 1).tolist()
        idx = 0
        for left, right in its.combinations(np.arange(self.m), 2):
            if self.mi[idx] > self.tau:
                subgroups = merge_set(subgroups, left, right)
            idx += 1
        print('\tAttribute partition: {}'.format(subgroups))
        self.attr_P = subgroups
        self.n_bins = len(self.attr_P)


    def make_dist_matrix(self,idx):
        bin = self.attr_P[idx]
        if len(bin) == 1:
            bin = bin[0]
            temp = self.data[bin]
            self.dist_rel_mat = torch.cdist(temp, temp, p=1).float()  # (n,d)-->>(n,n)
            if self.nominals[bin]:
                self.dist_rel_mat =  (self.dist_rel_mat > 1e-5).float()
        else:
            self.dist_rel_mat = torch.zeros(self.n, self.n, dtype=torch.float32).to(device)
            for j, bin_ in enumerate(bin):
                temp = self.data[bin_]
                mat = torch.cdist(temp, temp, p=1)
                if self.nominals[bin_]:
                    mat = (mat > 1e-5).float()
                self.dist_rel_mat += mat
            self.dist_rel_mat /= len(bin)


    def make_fuzzy_relation_mat(self, idx, p=0.8):
        self.make_dist_matrix(idx)
        # delta = 1 - torch.quantile(self.dist_rel_mat.view(-1), p).round(3)

        flattened = self.dist_rel_mat.view(-1)
        k = int(p * len(flattened))  # 第 k 小的值对应 p 分位数
        delta = 1 - (flattened.kthvalue(k).values * 1000).round()/1000


        self.dist_rel_mat *= -1
        self.dist_rel_mat += 1
        self.dist_rel_mat.masked_fill_(self.dist_rel_mat < delta, 0)
        # self.dist_rel_mat[self.dist_rel_mat < delta] = 0  # Winows+Pytorch 1.8.2 (cpu)实测，该赋值操作会额外占用大量内存
        self.weights[idx] = -torch.mean(torch.log2(self.dist_rel_mat.mean(dim=1))).unsqueeze(0)


    def fuzzy_granule_density(self, i):
        neighbors = (self.dist_rel_mat > 0).float()                    # N_i
        r_m_sum = self.dist_rel_mat.sum(dim=-1)                        # |[o_i]|
        self.FGD[i] = torch.square(r_m_sum) * (neighbors.mean(dim=-1)) # |N_i|/|O| * |[o_i]|^2
        self.FGD[i] /= torch.matmul(neighbors, r_m_sum)                # \sum_{o_j \in N_i} |[o_j]|

    def predict_score(self, p=None, data=None, nominals=None):
        if p is None and data is not None:  # Test mode
            p = self.p
            self.n, self.m = data.shape
            self.nominals = nominals
            self.data = torch.from_numpy(data.T).float().unsqueeze(-1).to(device)
            if self.mi is None:
                self.mutual_information(data)
                self.attribute_partition()


        self.FGD = torch.zeros((self.n_bins, self.n), dtype=torch.float32).to(device)
        self.weights = torch.zeros((self.n_bins, 1), dtype=torch.float32).to(device)
        for idx in range(self.n_bins):
            self.make_fuzzy_relation_mat(idx, p)
            self.fuzzy_granule_density(idx)
        od = 1 - (self.FGD * self.weights).mean(dim=0)
        del self.dist_rel_mat
        return od.cpu().numpy()
