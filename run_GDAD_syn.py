import os, time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import minmax_scale
from model5 import FAD

result_file_name = 'results_FAD5_syn.csv'
# open(result_file_name, 'w').write('dataset,model,strategy,threshold,lamb,auc,pr\n')

data_dir = 'data_syn/'
dataset_list = np.array(os.listdir(data_dir))

for dataset in dataset_list:
    d = np.load(data_dir+dataset)
    data = d['X']
    n, m = data.shape
    valids = np.std(data, axis=0) > 1e-6
    data = data[:, valids]
    label = d['y']
    nominals = d['nominals'][valids]
    print("{}\t\t shape:{}\t# Outlier:{}\t# Nominals:{}".format(dataset[:-4], (n, m), label.sum(),nominals.sum()))


    t0 = time.time()
    model = FAD(data, nominals)
    for threshold in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1][::-1]:
        repeated = model.make_dist_matrix(threshold=threshold)
        # if repeated:
        #     print('\tResults are same to former threshold...')
        #     continue
        for lamb in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            out_scores_ = model.predict_score(lamb=lamb)
            # print(model.dist_rel_mat.mean())

            for idx, out_scores in enumerate(out_scores_):
                auc = roc_auc_score(label, out_scores)
                pr = average_precision_score(y_true=label, y_score=out_scores, pos_label=1)
                print('\tLamb={}\tAUC={:.4f}\tPR={:.4f}'.format(lamb, auc, pr))

                scores = [dataset[:-4], 'FAD5', str(idx), str(threshold), str(lamb), str(auc)[:8], str(pr)[:8]]
                open(result_file_name, 'a').write(','.join(scores) + '\n')
        del model.dist_rel_mat

    t1 = time.time()
    print('\tTime={:.3f}'.format(t1-t0))
