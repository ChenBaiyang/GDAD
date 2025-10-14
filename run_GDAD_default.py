import os, time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_v1 import GDAD

result_file_name = 'results_GDAD_default.csv'
open(result_file_name, 'a').write('dataset,model,p,auc,pr\n')

data_dir = 'data/'
dataset_list = np.array(os.listdir(data_dir))

for dataset in dataset_list:
    d = np.load(data_dir+dataset)
    data, label, = d['X'], d['y']
    n, m = data.shape
    print("Dataset:{}\t\t shape:{}\t# Outlier:{}".format(dataset[:-4], (n, m), label.sum()))

    try:
        nominals = d['nominals']
    except:
        print("\tNominal attributes not designated, all attributes treated as numeric.")
        nominals = np.full(m, False, dtype=np.bool_)

    model = GDAD()
    out_scores = model.predict_score(data=data, nominals=nominals)

    # Evaluation
    auc = roc_auc_score(label, out_scores)
    pr = average_precision_score(y_true=label, y_score=out_scores, pos_label=1)
    print('\tResults: p={}\tAUC={:.4f}\tPR={:.4f}'.format(model.p, auc, pr))

    scores = [dataset[:-4], 'GDAD_default', str(model.p), str(auc)[:8], str(pr)[:8]]
    open(result_file_name, 'a').write(','.join(scores) + '\n')
