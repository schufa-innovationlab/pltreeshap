"""
This module defines the explainer class for computing SHAP (interaction) values of piecewise linear trees.
"""

#  Copyright 2022 SCHUFA Holding AG
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import numpy as np
from ._pltree import PLTree
from ._utils import convert_model


class PLTreeExplainer:
    """An explainer class using SHAP values to explain the output of piecewise linear tree ensembles.

    The interventional SHAP algorithms from [1]_ are implemented.
    Various regression models as well as binary classification models are supported.

    Parameters
    ----------
    model
        A predictive model which can be represented as an ensemble of piecewise linear trees. 
        Currently, the following models are supported:
        * Gradient Boosting Trees as implemented in LightGBM (https://github.com/microsoft/LightGBM). 
          This includes ensembles with piecewise linear trees (setting `linear_tree=True`; since LightGBM v3.3.0).
        * Gradient Boosting Trees (with piecewise constant trees) as implemented in XGBoost (https://github.com/dmlc/xgboost).
        * Model Trees as implemented in https://github.com/schufa-innovationlab/model-trees.
          Only  sklearn's LinearRegression and LogisticRegression are supported as leaf models.
    data : None or array-like of shape (num_data, num_features)
        An array holding a background dataset. If provided, split statistics for this dataset are computed (Algorithm 2 in [1]_), 
        which in turn are used for SHAP computations (i.e. Algorithm 3 in [1]_).
    
    Attributes
    ----------
    trees : list of PLTree
        The tree ensemble represented by a list of piecewise linear trees.

    References
    ----------
    .. [1] Zern, A. and Broelemann, K. and Kasneci, G.,
       "Interventional SHAP values and interaction values for Piecewise Linear Regression Trees",
       Proceedings of the AAAI Conference on Artificial Intelligence, 2023
    """

    def __init__(self, model, data=None):
        if not (isinstance(model, dict) and 'trees' in model):
            model = convert_model(model)
        self.trees = []
        for tree in model['trees']:
            coeffs = tree.get('coeffs', None)
            values = tree['values'].reshape((-1,)) if coeffs is None else tree['intercepts']
            decision_type = tree.get('decision_type', '<')
            pltree = PLTree(tree['children_left'], tree['children_right'], tree['children_default'], 
                            tree['features'], tree['thresholds'], values, coeffs, decision_type, data)
            self.trees.append(pltree)
    

    def aggregate(self, data):
        """Precompute split statistics for background data.

        The aggregation of a background dataset allows SHAP computations without iterating over that background dataset.
        This method implements Algorithm 2 from the paper [1]_.

        Parameters
        ----------
        data : array-like of shape (num_samples, num_features)
            An array holding the background dataset for SHAP computation.
        """

        data = np.asarray(data, dtype=np.float64)
        for tree in self.trees:
            tree.aggregate(data)

    
    def predict(self, x):
        """Computes the prediction of the piecewise linear tree ensemble for given input `x`.

        Parameters
        ----------
        x : array-like of shape (num_samples, num_features)
            An array holding the input data for prediction.
        
        Returns
        -------
        out : array of shape (num_samples,)
            An array containing the predictions for each input sample.
        """

        x = np.asarray(x, dtype=np.float64)
        out = np.zeros((x.shape[0],))
        for tree in self.trees:
            out = tree.predict(x, out)
        return out


    def shap_values(self, x, data=None):
        """Computes the (interventional) SHAP values for samples `x`.

        If background data (`data`) is provided, the SHAP values are computed by iterating over the
        background data points (Algorithm 1 in the paper [1]_). Otherwise precomputed split statistics
        are used for SHAP computation (Algorithm 3 in the paper [1]_).

        Parameters
        ----------
        x : array-like of shape (num_samples, num_features)
            Sample points whose prediction should be explained by SHAP values.
        data : None or array-like of shape (num_data, num_features)
            Background data for SHAP computation. These data is needed if no split statistics were 
            precomputed (using the method `aggregate`).
        
        Returns
        -------
        phi : array of shape (num_samples, num_features)
            Array containing the SHAP values for samples `x`.
        """

        x = np.asarray(x, dtype=np.float64)
        assert x.ndim == 2
        if data is not None:
            data = np.asarray(data, dtype=np.float64)
        
        phi = np.zeros(x.shape, dtype=np.float64)
        for tree in self.trees:
            phi = tree.shap_values(x, data, phi)
        return phi


    def shap_interaction_values(self, x, data=None):
        """Computes the (interventional) SHAP interaction values for samples `x`.

        If background data (`data`) is provided, the SHAP interaction values are computed by iterating over the
        background data points (Algorithm 1 in the paper [1]_). Otherwise precomputed split statistics
        are used for SHAP computation (Algorithm 3 in the paper [1]_).
        The output array holds matrices phi[k,:,:] for each sample point x[k,:]. The nondiagonal elements
        are the corresponding interaction values scaled by 0.5, i.e. phi[k,i,j] = 0.5 * phi_{i,j}(x[k,:]).
        The diagonal elements store the corresponding SHAP values corrected by the interaction values, i.e.
        phi[k,i,i] = phi_{i}(x[k,:]) - 0.5 * sum_{j!=i} phi_{i,j}(x[k,:]).

        Parameters
        ----------
        x : array-like of shape (num_samples, num_features)
            Sample points whose prediction should be explained by SHAP interaction values.
        data : None or array-like of shape (num_data, num_features)
            Background data for SHAP computation. These data is needed if no split statistics were 
            precomputed (using the method `aggregate`).
            
        Returns
        -------
        phi : array of shape (num_samples, num_features, num_features)
            Array containing the SHAP interaction values for samples `x`.
        """

        x = np.asarray(x, dtype=np.float64)
        assert x.ndim == 2
        if data is not None:
            data = np.asarray(data, dtype=np.float64)

        phi = np.zeros((*x.shape, x.shape[1]), dtype=np.float64)
        for tree in self.trees:
            phi = tree.shap_interaction_values(x, data, phi)
        return phi
