import os, time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import minmax_scale
from model import GDAD

'''
A previous version of the GDAD codebase was lost due to a BitLocker encryption issue. 
This release represents a complete reconstruction of the project. The rebuild provided 
an opportunity to implement significant improvements, most notably the new GPU 
acceleration support. While the core algorithm logic has been carefully recreated from 
the original paper, the numerical results may exhibit slight variations compared to those
 reported in the published paper.
'''