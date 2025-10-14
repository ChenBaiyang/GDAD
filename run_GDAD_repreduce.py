import os, time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import minmax_scale
from model import GDAD

result_file_name = 'results_GDAD.csv'
open(result_file_name, 'a').write('dataset,model,tau,p,auc,pr\n')

data_dir = 'data/'
dataset_list = np.array(os.listdir(data_dir))[22:]

for dataset in dataset_list:
    d = np.load(data_dir+dataset)
    data, label, nominals = d['X'], d['y'], d['nominals']
    n, m = data.shape
    print("{}\t\t shape:{}\t# Outlier:{}\t# Nominals:{}".format(dataset[:-4], (n, m), label.sum(),nominals.sum()))

    if n > 7000 or m > 100:
        continue

    model = GDAD(data, nominals)
    paras = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for tau in paras:
        model.make_dist_matrix(tau=tau)
        for p in paras:
            out_scores = model.predict_score(p=p)

            auc = roc_auc_score(label, out_scores)
            pr = average_precision_score(y_true=label, y_score=out_scores, pos_label=1)
            print('\tp={}\tAUC={:.4f}\tPR={:.4f}'.format(p, auc, pr))

            scores = [dataset[:-4], 'GDAD', str(tau), str(p), str(auc)[:8], str(pr)[:8]]
            open(result_file_name, 'a').write(','.join(scores) + '\n')

    # break