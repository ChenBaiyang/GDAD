# GDAD
Baiyang Chen, Zhong Yuan, Dezhong Peng, Xiaoliang Chen, Hongmei Chen, Yingke Chen: "[Integrating granular computing with density estimation for anomaly detection in high-dimensional heterogeneous data](https://www.sciencedirect.com/science/article/pii/S0020025524014804)"[J]. Information sciences, 690: 121566 (2025), DOI: 10.1016/j.ins.2024.121566

## Abstract
Outlier detection aims to find objects that behave differently from the majority of the data. Existing unsupervised approaches often process data with a single scale, which may not capture the multi-scale nature of the data. In this paper, we propose a novel information fusion model based on multi-scale fuzzy granules and an unsupervised outlier detection algorithm with the fuzzy rough set theory. First, a multi-scale information fusion model is formulated based on fuzzy granules. Then we employ fuzzy approximations to define the outlier factor of multi-scale fuzzy granules centered at each data point. Finally, the outlier score is calculated by aggregating the outlier factors of a set of multi-scale fuzzy granules.

## Citation
If you find the code or datasets useful in your research, please consider citing:
```
@Article{chen2024GDAD,
  author  = {Baiyang Chen and Zhong Yuan and Dezhong Peng and Xiaoliang Chen and Hongmei Chen and Yingke Chen},
  journal = {Information Sciences},
  title   = {Integrating granular computing with density estimation for anomaly detection in high-dimensional heterogeneous data},
  year    = {2025},
  issn    = {0020-0255},
  pages   = {121566},
  volume  = {690},
  doi     = {10.1016/j.ins.2024.121566},
}
```

## Environment
* python=3.8
* pytorch=1.8.2
* scikit-learn=1.2

## Usage
Assume the dataset be saved in a Numpy npz file with n samples and m dimensions. An m dimensional bool vector be given to indicate: True=Nominal attribute, False=Numerical attribute; if not provided, all attributes are treated as numerical.

To run GDAD with default parameters:
```
python run_GDAD_GridSearch.py
```
To run GDAD with parameter tuning:
```
python run_GDAD_default.py
```

## Contact
If you have any question, please contact farstars@qq.com