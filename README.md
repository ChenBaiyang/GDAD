# GDAD
Baiyang Chen, Zhong Yuan, Dezhong Peng, Xiaoliang Chen, Hongmei Chen, Yingke Chen: "[Integrating granular computing with density estimation for anomaly detection in high-dimensional heterogeneous data](https://www.sciencedirect.com/science/article/pii/S0020025524014804)"[J]. Information sciences, 690: 121566 (2025), DOI: 10.1016/j.ins.2024.121566

## Abstract
Detecting anomalies in complex data is crucial for knowledge discovery and data mining
across a wide range of applications. While density-based methods are effective for handling
varying data densities and diverse distributions, they often struggle with accurately estimating
densities in heterogeneous, uncertain data and capturing interdependencies among features in
high-dimensional spaces. This paper proposes a fuzzy granule density-based anomaly detection
algorithm (GDAD) for heterogeneous data. Specifically, GDAD first partitions high-dimensional
attributes into subspaces based on their interdependencies and employs fuzzy information
granules to represent data. The core of the method is the definition of fuzzy granule density,
which leverages local neighborhood information alongside global density patterns and effectively
characterizes anomalies in data. Each object is then assigned a fuzzy granule density-based
anomaly factor, reflecting its likelihood of being anomalous. Through extensive experimentation
on various real-world datasets, GDAD has demonstrated superior performance, matching or
surpassing existing state-of-the-art methods. GDAD’s integration of granular computing with
density estimation provides a practical framework for anomaly detection in high-dimensional
heterogeneous data.

## Environment
* python=3.8
* pytorch=1.8.2
* scikit-learn=1.2

## Usage
Assume the dataset be saved in a Numpy npz file with n samples and m dimensions. An m dimensional bool vector be given to indicate: True=Nominal attribute, False=Numerical attribute; if not provided, all attributes are treated as numerical.

To run GDAD with default parameters:
```
python run_GDAD_default.py
```
To run GDAD with parameter tuning:

We only tune the fuzzy radius parameter $p$ (percentile) among the set of {0.1, 0.3, 0.5, 0.7, 0.9}.
```
python run_GDAD_GridSearch.py
```


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

## Contact
If you have any question, please contact farstars@qq.com