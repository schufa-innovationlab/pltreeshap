# Interventional SHAP for Piecewise Linear Trees
This package implements algorithms for computing interventional SHAP values and interaction values of piecewise linear trees and their ensembles.
This includes piecewise linear tree ensembles from [LightGBM](https://github.com/microsoft/LightGBM) (`LGBMRegressor(linear_tree=True)`, `LGBMClassifier(linear_tree=True)`).

The implemented algorithms are described in the paper [[1]](#References).

## Installation
The package can directly be installed from GitHub using pip:
```shell script
pip install https://github.com/schufa-innovationlab/pltreeshap/archive/main.zip
```
For this, an installed C++ compiler is required (e.g. Microsoft Visual C++ 14.0 for Windows).


## Usage
The package provides the explainer class `PLTreeExplainer` implementing the methods `shap_values` and `shap_interaction_values`.

### Introductory Example
```python
from pltreeshap import PLTreeExplainer

# get and train model
model = ...

# set up explainer with background data (e.g. training data)
explainer = PLTreeExplainer(model, data=data)

# compute SHAP values of some sample points (e.g. validation data)
phi = explainer.shap_values(x)

# compute SHAP interaction values
phi2 = explainer.shap_interaction_values(x)
```

### Iterating over Background Data vs. Using Precomputed Split Statistics
Both SHAP algorithms from the paper [[1]](#References) are implemented. The first one iterates over the background dataset. This algorithm is used with the following code.
```python
explainer = PLTreeExplainer(model)
phi = explainer.shap_values(x, data=data)  # iteration over points in `data`
```

The second variant precomputes split statistics of the background dataset and uses this statistics for SHAP computation. The statistics are precomputed when passing the data to the explainer class, i.e. `PLTreeExplainer(model, data=data)`, or by the following code.
```python
explainer = PLTreeExplainer(model)
explainer.aggregate(data)  # precomputes split statistics
phi = explainer.shap_values(x)
```

## References
[1] Zern, A. and Broelemann, K. and Kasneci, G.;
Interventional SHAP Values and Interaction Values for Piecewise Linear Regression Trees;
Proceedings of the AAAI Conference on Artificial Intelligence, 2023;
<details><summary>Bibtex</summary>
<p>

```
@inproceedings{Zern2023Interventional,
    author = {Artjom Zern and Klaus Broelemann and Gjergji Kasneci},
    title  = {Interventional SHAP Values and Interaction Values for Piecewise Linear Regression Trees},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year = {2023}
}
```

</p>
</details>