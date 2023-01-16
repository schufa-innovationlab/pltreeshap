# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False


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
cimport numpy as cnp
from libc.math cimport isnan
from libcpp.vector cimport vector as cpp_vector
from scipy.special.cython_special cimport binom

ctypedef cnp.uint8_t BOOL_t
ctypedef cnp.intp_t SIZE_t
ctypedef int NODE_t
ctypedef int FEATURE_t
# define the signature of functions used to check split condition within the tree:
ctypedef bint (*cmp_func)(double, double, bint) nogil
# define the function signature of Shapley updates (given by functions with names shapley_{output}_{input} defined below):
ctypedef void (*shapley_of_interval)(double, cpp_vector[double]&, cpp_vector[FEATURE_t]&, cpp_vector[FEATURE_t]&, double*)

# define function used for split conditions (i.e. if output is `True` then the left child node is taken):
cdef inline bint less(double x, double t, bint default) nogil:
    return (isnan(x) and default) or (not isnan(x) and x < t)

cdef inline bint less_equal(double x, double t, bint default) nogil:
    return (isnan(x) and default) or (not isnan(x) and x <= t)

cdef inline bint greater(double x, double t, bint default) nogil:
    return (isnan(x) and default) or (not isnan(x) and x > t)

cdef inline bint greater_equal(double x, double t, bint default) nogil:
    return (isnan(x) and default) or (not isnan(x) and x >= t)


# define functions for updating Shapley (interaction) values used for SHAP computations
cdef void shapley_values_const(double value, cpp_vector[double]& coeffs,
                               cpp_vector[FEATURE_t]& A, cpp_vector[FEATURE_t]& NB, double* phi):
    """Updates `phi` by the Shapley values of v*1_[A,B], where v=`value` is a constant coalition. 
    
    The set B is given as complement of `NB`. `coeffs` is ignored, since this function considers only constant coalition functions.
    """

    cdef:
        SIZE_t nA = A.size()
        SIZE_t nNB = NB.size()
        FEATURE_t j
        double phi_update
    if nA > 0:
        phi_update = value / ((nA + nNB) * binom(nA + nNB - 1, nNB))
        for j in A:
            phi[j] += phi_update
    if nNB > 0:
        phi_update = -value / ((nA + nNB) * binom(nA + nNB - 1, nA))
        for j in NB:
            phi[j] += phi_update


cdef void shapley_values_linear(double value, cpp_vector[double]& coeffs,
                                cpp_vector[FEATURE_t]& A, cpp_vector[FEATURE_t]& NB, double* phi):
    """Updates `phi` by the Shapley values of v*1_[A,B], where v is a linear coalition given by `value` and `coeffs`."""

    cdef:
        SIZE_t nA = A.size()
        SIZE_t nNB = NB.size()
        SIZE_t num_features = coeffs.size()
        FEATURE_t j
        cpp_vector[BOOL_t] feature_used = cpp_vector[BOOL_t](num_features, 0)
        cpp_vector[FEATURE_t] BA = cpp_vector[FEATURE_t]()
        double sum_A, sum_BA, phi_update
    BA.reserve(num_features - nA - nNB)
    sum_BA = 0.
    for j in A:
        feature_used[j] = 1
    for j in NB:
        feature_used[j] = 1
    for j in range(num_features):
        if feature_used[j] == 0:
            BA.push_back(j)
            sum_BA += coeffs[j]
    sum_A = value
    for j in A:
        sum_A += coeffs[j]
    if nA > 0:
        phi_update = sum_A / ((nA + nNB) * binom(nA + nNB - 1, nNB)) \
                     + sum_BA / ((nA + nNB + 1) * binom(nA + nNB, nNB))
        for j in A:
            phi[j] += phi_update
    if nNB > 0:
        phi_update = -sum_A / ((nA + nNB) * binom(nA + nNB - 1, nA)) \
                     - sum_BA / ((nA + nNB + 1) * binom(nA + nNB, nA + 1))
        for j in NB:
            phi[j] += phi_update
    phi_update = 1. / ((nA + nNB + 1) * binom(nA + nNB, nA))
    for j in BA:
        phi[j] += coeffs[j] * phi_update


cdef void shapley_interaction_const(double value, cpp_vector[double]& coeffs,
                                    cpp_vector[FEATURE_t]& A, cpp_vector[FEATURE_t]& NB, double* phi):
    """Updates `phi` by the Shapley interaction values of v*1_[A,B], where v=`value` is a constant coalition.
    
    `phi` is assumed to be flattened, i.e. phi[i,j] is given by phi[num_features*i+j].
    """

    cdef:
        SIZE_t nA = A.size()
        SIZE_t nNB = NB.size()
        SIZE_t num_features = coeffs.size()  # = original dimensions of `phi`
        FEATURE_t j, l
        double phi_update
    
    # update SHAP values on the diagonal:
    if nA > 0:
        phi_update = value / ((nA + nNB) * binom(nA + nNB - 1, nNB))
        for j in A:
            phi[num_features*j+j] += phi_update
    if nNB > 0:
        phi_update = -value / ((nA + nNB) * binom(nA + nNB - 1, nA))
        for j in NB:
            phi[num_features*j+j] += phi_update
    
    # update SHAP interaction values (with a factor 0.5):
    if nA >= 2:
        phi_update = 0.5 * value / ((nA + nNB - 1) * binom(nA + nNB - 2, nNB))
        for j in A:
            for l in A:
                if l == j:
                    continue
                phi[num_features*j+l] += phi_update  # update interaction values
                phi[num_features*j+j] -= phi_update  # correct diagonal
    if nA >= 1 and nNB >= 1:
        phi_update = -0.5 * value / ((nA + nNB - 1) * binom(nA + nNB - 2, nNB - 1))
        for j in A:
            for l in NB:
                phi[num_features*j+l] += phi_update  # update interaction values
                phi[num_features*l+j] += phi_update
                phi[num_features*j+j] -= phi_update  # correct diagonal
                phi[num_features*l+l] -= phi_update
    if nNB >= 2:
        phi_update = 0.5 * value / ((nA + nNB - 1) * binom(nA + nNB - 2, nA))
        for j in NB:
            for l in NB:
                if l == j:
                    continue
                phi[num_features*j+l] += phi_update  # update interaction values
                phi[num_features*j+j] -= phi_update  # correct diagonal


cdef void shapley_interaction_linear(double value, cpp_vector[double]& coeffs,
                                     cpp_vector[FEATURE_t]& A, cpp_vector[FEATURE_t]& NB, double* phi):
    """Updates `phi` by the Shapley interaction values of v*1_[A,B], where v is a linear coalition given by `value` and `coeffs`.
    
    `phi` is assumed to be flattened, i.e. phi[i,j] is given by phi[num_features*i+j].
    """

    cdef:
        SIZE_t nA = A.size()
        SIZE_t nNB = NB.size()
        SIZE_t num_features = coeffs.size()
        FEATURE_t j, l
        cpp_vector[BOOL_t] feature_used = cpp_vector[BOOL_t](num_features, 0)
        cpp_vector[FEATURE_t] BA = cpp_vector[FEATURE_t]()
        double sum_A, sum_BA, phi_update
    
    # precompute B\A and the sums over A and B\A:
    BA.reserve(num_features - nA - nNB)
    sum_BA = 0.
    for j in A:
        feature_used[j] = 1
    for j in NB:
        feature_used[j] = 1
    for j in range(num_features):
        if feature_used[j] == 0:
            BA.push_back(j)
            sum_BA += coeffs[j]
    sum_A = value
    for j in A:
        sum_A += coeffs[j]
    
    # update SHAP values on the diagonal:
    if nA > 0:
        phi_update = sum_A / ((nA + nNB) * binom(nA + nNB - 1, nNB)) \
                     + sum_BA / ((nA + nNB + 1) * binom(nA + nNB, nNB))
        for j in A:
            phi[num_features*j+j] += phi_update
    if nNB > 0:
        phi_update = -sum_A / ((nA + nNB) * binom(nA + nNB - 1, nA)) \
                     - sum_BA / ((nA + nNB + 1) * binom(nA + nNB, nA + 1))
        for j in NB:
            phi[num_features*j+j] += phi_update
    phi_update = 1. / ((nA + nNB + 1) * binom(nA + nNB, nA))
    for j in BA:
        phi[num_features*j+j] += coeffs[j] * phi_update
    
    # update SHAP interaction values (with a factor 0.5):
    if nA >= 2:
        phi_update = 0.5 * ( sum_A / ((nA + nNB - 1) * binom(nA + nNB - 2, nNB))
                             + sum_BA / ((nA + nNB) * binom(nA + nNB - 1, nNB)) )
        for j in A:
            for l in A:
                if l == j:
                    continue
                phi[num_features*j+l] += phi_update  # update interaction values
                phi[num_features*j+j] -= phi_update  # correct diagonal
    if nA >= 1 and nNB >= 1:
        phi_update = -0.5 * ( sum_A / ((nA + nNB - 1) * binom(nA + nNB - 2, nA - 1))
                              + sum_BA / ((nA + nNB) * binom(nA + nNB - 1, nA)) )
        for j in A:
            for l in NB:
                phi[num_features*j+l] += phi_update  # update interaction values
                phi[num_features*l+j] += phi_update
                phi[num_features*j+j] -= phi_update  # correct diagonal
                phi[num_features*l+l] -= phi_update
    if nNB >= 2:
        phi_update = 0.5 * ( sum_A / ((nA + nNB - 1) * binom(nA + nNB - 2, nA)) 
                             + sum_BA / ((nA + nNB) * binom(nA + nNB - 1, nA + 1)) )
        for j in NB:
            for l in NB:
                if l == j:
                    continue
                phi[num_features*j+l] += phi_update  # update interaction values
                phi[num_features*j+j] -= phi_update  # correct diagonal
    if nA >= 1:
        phi_update = 0.5 / ((nA + nNB) * binom(nA + nNB - 1, nNB))
        for j in A:
            for l in BA:
                phi[num_features*j+l] += coeffs[l] * phi_update  # update interaction values
                phi[num_features*l+j] += coeffs[l] * phi_update
                phi[num_features*j+j] -= coeffs[l] * phi_update  # correct diagonal
                phi[num_features*l+l] -= coeffs[l] * phi_update
    if nNB >= 1:
        phi_update = -0.5 / ((nA + nNB) * binom(nA + nNB - 1, nA))
        for j in NB:
            for l in BA:
                phi[num_features*j+l] += coeffs[l] * phi_update  # update interaction values
                phi[num_features*l+j] += coeffs[l] * phi_update
                phi[num_features*j+j] -= coeffs[l] * phi_update  # correct diagonal
                phi[num_features*l+l] -= coeffs[l] * phi_update
    
    
cdef class PLTree:
    """A piecewise linear tree.

    This class also provides methods for computing SHAP (interaction) values. For this, algorithms from [1]_ are implemented.

    Parameters
    ----------
    child_left : array-like of shape (num_nodes,)
        Array containing the indices of the left child nodes. For leaf nodes these indices are -1.
    child_right : array-like of shape (num_nodes,)
        Array containing the indices of the right child nodes. For leaf nodes these indices are -1.
    child_default : array-like of shape (num_nodes,)
        Array containing the indices of the left child nodes. For leaf nodes these indices are -1.
    split_feature : array-like of shape (num_nodes,)
        Array containing the indices of features, which are used for splitting at the corresponding node.
        For leaf nodes these indices are -1.
    threshold : array-like of shape (num_nodes,)
        Array of splitting thresholds.
    value : array-like of shape (num_nodes,)
        In case of piecewise constant trees (`coeffs` is None), this array contains the values for every node. 
        For piecewise linear trees, `value` holds the intercepts of the linear models at each node.
    coeffs : None or array-like of shape (num_nodes, num_features)
        If `coeffs` is None, then the tree is assumed to be piecewise constant with values given by `value`.
        Otherwise `coeffs` is assumed to hold the linear coefficients of the linear model for each node.
    decision_type : '<' or '<=' or '>' or '>='
        The decition type of the tree. For instance, if decision_type=='<', then the left child node is taken if
        x[feature[node]] < threshold[node].
    data : None or array-like of shape (num_data, num_features)
        The background data. If it is provided, then split statistics are computed for these data (Algorithm 2 in [1]_). 
        These precomputed statistics are used in SHAP computations (Algorithm 3 in [1]_).
    
    Attributes
    ----------
    num_nodes : int
        Number of nodes.
    num_nodes_agg : int
        Number of nodes of the tree storing the split statistics of the background dataset.
        If no background data was aggregated, then this value is -1.
    num_features : int
        Number of features of the data. If the tree is piecewise constant, 
        then `num_features` is the minimal number of features.
    max_depth : int
        The depth of the tree.
    is_linear : bool
        Indicates whether the tree is piecewise linear or piecewise constant.
    has_aggregated : bool
        Whether background data has been aggregated or not.
    child_left : array of shape (num_nodes,)
        Array containing the indices of the left child nodes.
    child_right : array of shape (num_nodes,)
        Array containing the indices of the right child nodes.
    left_is_default : array of shape (num_nodes,)
        Array indicating whether the left child node is the default node (left_is_default[node]==1) 
        or not (left_is_default[node]==0).
    split_feature : array of shape (num_nodes,)
        Array holding the splitting features for each node.
    threshold : array of shape (num_nodes,)
        Array holding the splitting thresholds for each node.
    value : array of shape (num_nodes,)
        Array holding the values (in case of piecewise constant tree) or the intercepts (in case of 
        piecewise linear tree) for each node.
    coeffs : None or array of shape (num_nodes, num_features)
        In case of piecewise linear tree, `coeffs` holds the linear coefficients for each node.

    References
    ----------
    .. [1] Zern, A. and Broelemann, K. and Kasneci, G.,
       "Interventional SHAP values and interaction values for Piecewise Linear Regression Trees",
       Proceedings of the AAAI Conference on Artificial Intelligence, 2023
    """

    cdef:
        readonly SIZE_t num_nodes  # number of nodes of the tree
        readonly SIZE_t num_nodes_agg  # number of nodes of the tree used for storing aggregated data
        readonly SIZE_t num_features  # number of features
        readonly SIZE_t max_depth  # depth of the tree
        readonly bint is_linear  # whether nodes hold linear models (is_linear==True) or a constant
        readonly bint has_aggregated  # whether the tree for storing aggregated data was constructed
        NODE_t[::1] child_left  # indices of left child nodes
        NODE_t[::1] child_right  # indices of right child nodes
        BOOL_t[::1] left_is_default  # whether the left child node is the default node
        NODE_t[::1] parent  # indices of parent nodes
        NODE_t[::1] parent_agg  # indices of parent nodes in the tree for storing aggregated data
        NODE_t[::1] ancestor_same_feature  # indices of ancestor nodes with same split features (see `__cinit__`)
        NODE_t[::1] ancestor_agg_same_feature  # indices of ancestor nodes in the tree for storing aggregated data (see `create_agg_tree`)
        NODE_t[::1] child_left_true  # indices of child nodes of the tree for storing aggregated data (see `create_agg_tree`)
        NODE_t[::1] child_left_false
        NODE_t[::1] child_right_true
        NODE_t[::1] child_right_false
        int[::1] depth  # the depth of each node
        NODE_t[::1] node_orig  # maps of nodes of the tree holding aggregated data to nodes of the original tree
        FEATURE_t[::1] split_feature
        double[::1] threshold  # thresholds for splitting
        double[::1] value  # constants (or intercepts) at each node
        double[:,::1] coeffs  # linear coefficients of each node
        double[::1] cover  # node cover of the tree storing the aggregated data
        double[:,::1] mean_data  # aggregated data (stored in a new tree structure)
        cmp_func take_left_node  # decision function deciding whether the left node should be taken
    

    def __cinit__(self, child_left, child_right, child_default, split_feature, threshold, value, coeffs=None,
                  decicsion_type='<', data=None):
        cdef:
            NODE_t node, node0, child
            cpp_vector[NODE_t] stack = cpp_vector[NODE_t]()
        
        if decicsion_type == '<':
            self.take_left_node = less
        elif decicsion_type == '<=':
            self.take_left_node = less_equal
        elif decicsion_type == '>':
            self.take_left_node = greater
        elif decicsion_type == '>=':
            self.take_left_node = greater_equal
        else:
            raise ValueError("`decicsion_type` should be one of the four strings: '<', '<=', '>', '>='")
        
        assert len(child_left) == len(child_right), "All arrays should have same length!"
        assert len(child_left) == len(child_default), "All arrays should have same length!"
        assert len(child_left) == len(split_feature), "All arrays should have same length!"
        assert len(child_left) == len(threshold), "All arrays should have same length!"
        assert len(child_left) == len(value), "All arrays should have same length!"
        if coeffs is not None:
            assert len(child_left) == len(coeffs), "All arrays should have same length!"
        
        self.num_nodes = len(child_left)
        self.child_left = np.array(child_left, dtype=np.intc)
        self.child_right = np.array(child_right, dtype=np.intc)
        self.split_feature = np.array(split_feature, dtype=np.intc)
        self.threshold = np.array(threshold, dtype=np.float64)
        self.value = np.array(value, dtype=np.float64)
        self.left_is_default = (self.child_left == np.asarray(child_default))
        
        if coeffs is not None:
            self.is_linear = True
            self.coeffs = np.array(coeffs, dtype=np.float64)
            self.num_features = self.coeffs.shape[1]
        else:
            self.is_linear = False
            self.coeffs = None

        if not self.is_linear:
            self.num_features = np.amax(split_feature) + 1
        
        self.depth = np.zeros((self.num_nodes,), dtype=np.intc)
        
        self.max_depth = 0
        stack.push_back(0)
        while not stack.empty():
            node = stack.back()
            stack.pop_back()
            child = self.child_left[node]
            if child >= 0:
                self.depth[child] = self.depth[node] + 1
                stack.push_back(child)
            else:
                self.max_depth = max(self.depth[node], self.max_depth)
            child = self.child_right[node]
            if child >= 0:
                self.depth[child] = self.depth[node] + 1
                stack.push_back(child)
            else:
                self.max_depth = max(self.depth[node], self.max_depth)
        
        self.parent = np.full((self.num_nodes,), -1, dtype=np.intc)
        for node in range(self.num_nodes):
            if self.child_left[node] >= 0:
                self.parent[self.child_left[node]] = node
            if self.child_right[node] >= 0:
                self.parent[self.child_right[node]] = node
        # In the following, `ancestor_same_feature` is computed. This array stores for each node the next ancestor node, 
        # whose parent has the same split feature. It is used in the method `shap_mean_baseline` to check if a feature
        # was already used on the decision path.
        self.ancestor_same_feature = np.full((self.num_nodes,), -1, dtype=np.intc)
        for node0 in range(self.num_nodes):
            if self.split_feature[node0] < 0:
                continue
            child = node0
            node = self.parent[child]
            while node >= 0:
                if self.split_feature[node] == self.split_feature[node0]:
                    self.ancestor_same_feature[node0] = child
                    break
                child = node
                node = self.parent[child]
        
        self.has_aggregated = False
        self.num_nodes_agg = -1
        self.cover = None
        self.mean_data = None
        self.node_orig = None
        self.parent_agg = None
        self.ancestor_agg_same_feature = None
        if data is not None:
            self.aggregate(data)
    
    
    @property
    def child_left(self):
        return np.asarray(self.child_left)
    
    @property
    def child_right(self):
        return np.asarray(self.child_right)
    
    @property
    def left_is_default(self):
        return np.asarray(self.left_is_default)
    
    @property
    def split_feature(self):
        return np.asarray(self.split_feature)
    
    @property
    def threshold(self):
        return np.asarray(self.threshold)
    
    @property
    def value(self):
        return np.asarray(self.value)
    
    @property
    def coeffs(self):
        if self.coeffs is not None:
            return np.asarray(self.coeffs)
        else: 
            return None
    
    
    cdef void create_agg_tree(self):
        """Creates a tree for storing split statistics of background data.
        
        The new tree structure is derived from the tree structure of the original piecewise linear tree.
        For every node in the original tree, the new tree contains 2^depth(node) copies of this node.
        Every inner node in the new tree has four child nodes (`child_left_true`, `child_left_false`, 
        `child_right_true`, `child_right_false`).
        """

        cdef:
            NODE_t node0, child0  # `node0` and `child0` are nodes in the original tree
            NODE_t node, child  # `node` and `child` are nodes in the new tree
            cpp_vector[NODE_t] indices = cpp_vector[NODE_t](self.num_nodes + 1)
            cpp_vector[NODE_t] stack = cpp_vector[NODE_t]()
            FEATURE_t j, l
        # Every node `node0` in the original tree has 2^depth(node0) copies in the new tree structure.
        # The indices of those copies range from indices[node0] to indices[node0 + 1] - 1.
        indices[0] = 0
        for node0 in range(self.num_nodes):
            indices[node0 + 1] = indices[node0] + (1 << self.depth[node0])
        self.num_nodes_agg = indices[self.num_nodes]  # = number of nodes in the new tree structure

        self.child_left_true = np.full((self.num_nodes_agg,), -1, dtype=np.intc)
        self.child_left_false = np.full((self.num_nodes_agg,), -1, dtype=np.intc)
        self.child_right_true = np.full((self.num_nodes_agg,), -1, dtype=np.intc)
        self.child_right_false = np.full((self.num_nodes_agg,), -1, dtype=np.intc)
        self.parent_agg = np.full((self.num_nodes_agg,), -1, dtype=np.intc)  # indices of parent nodes in the new tree
        self.node_orig = np.empty((self.num_nodes_agg,), dtype=np.intc)

        for node0 in range(self.num_nodes):
            for node in range(indices[node0], indices[node0+1]):
                self.node_orig[node] = node0  # map nodes in the new tree to corresponding nodes in the orignal tree

        # Compute child indices by (arbitrary) distributing node indices such that the structure of the new tree is 
        # compatible with the original tree, i.e. node_orig[child_left_true[node]] = child_left[node_orig[node]].
        stack.push_back(0)
        while not stack.empty():
            node0 = stack.back()
            stack.pop_back()
            child0 = self.child_left[node0]
            if child0 >= 0:
                child = indices[child0]
                for node in range(indices[node0], indices[node0 + 1]):
                    self.child_left_true[node] = child
                    self.child_left_false[node] = child + 1
                    self.parent_agg[child] = node
                    self.parent_agg[child + 1] = node
                    child += 2
                stack.push_back(child0)
            child0 = self.child_right[node0]
            if child0 >= 0:
                child = indices[child0]
                for node in range(indices[node0], indices[node0 + 1]):
                    self.child_right_true[node] = child
                    self.child_right_false[node] = child + 1
                    self.parent_agg[child] = node
                    self.parent_agg[child + 1] = node
                    child += 2
                stack.push_back(child0)
        
        # In the following `ancestor_agg_same_feature` is computed. For each node in the new tree structure, this array stores,
        # the index of the next ancestor node, whose parent has the same split feature. It is used in the method `shap_aggregated`
        # to check if a feature was already used on the decision path.
        self.ancestor_agg_same_feature = np.full((self.num_nodes_agg,), -1, dtype=np.intc)
        for child in range(self.num_nodes_agg):
            child0 = self.node_orig[child]
            if self.ancestor_same_feature[child0] >= 0:
                node = child
                while node >= 0:
                    if self.ancestor_same_feature[child0] == self.node_orig[node]:
                        self.ancestor_agg_same_feature[child] = node
                        break
                    node = self.parent_agg[node]

    
    cpdef void aggregate(self, double[:,:] data):
        """Computes split statistics for given background data (Algorithm 2 in the paper [1]_).
        
        Parameters
        ----------
        data : array of shape (num_data, num_features)
            An array representing the background dataset for SHAP computation.
        """

        cdef:
            SIZE_t num_data = data.shape[0]
            NODE_t node, node0
            cpp_vector[NODE_t] indices
            cpp_vector[NODE_t] stack = cpp_vector[NODE_t]()
            SIZE_t k
            FEATURE_t j, l
        if self.is_linear:
            assert data.shape[1] == self.num_features
        else:
            assert data.shape[1] >= self.num_features
        stack.reserve(self.num_nodes)
        
        # child_left_true, child_right_true, child_left_false, child_right_false, and node_orig are independent of
        # the data. If some data have been aggregated, then these arrays do not have to be (re-)computed.
        if not self.has_aggregated:
            self.create_agg_tree()
        
        self.cover = np.zeros((self.num_nodes_agg,))
        if self.is_linear:
            self.mean_data = np.zeros((self.num_nodes_agg, self.num_features))
        for k in range(num_data):
            stack.push_back(0)
            while not stack.empty():
                node = stack.back()
                stack.pop_back()
                self.cover[node] += 1. / num_data
                if self.is_linear:
                    for l in range(self.num_features):
                        # online update of the mean vectors:
                        self.mean_data[node,l] += (data[k,l] - self.mean_data[node,l]) / (num_data * self.cover[node])
                node0 = self.node_orig[node]
                j = self.split_feature[node0]
                if j >= 0:
                    if self.take_left_node(data[k,j], self.threshold[node0], self.left_is_default[node0]):
                        stack.push_back(self.child_left_true[node])
                        stack.push_back(self.child_right_false[node])
                    else:
                        stack.push_back(self.child_right_true[node])
                        stack.push_back(self.child_left_false[node])
        self.has_aggregated = True
    
    
    def predict(self, double[:,:] x, double[:] out=None):
        """Computes the prediction of the piecewise linear tree for given input `x`.
        
        Parameters
        ----------
        x : array of shape (num_samples, num_features)
            Input data.
        out : None or array of shape (num_samples,)
            Output array. If provided, then the prediction is added to this array. This is used for prediction of ensembles.

        Returns
        -------
        out : array of shape (num_samples,)
            Predicted values.
        """

        cdef:
            SIZE_t num_samples = x.shape[0]
            SIZE_t k
            NODE_t node
            FEATURE_t j
        if self.is_linear:
            assert x.shape[1] == self.num_features
        else:
            assert x.shape[1] >= self.num_features
        
        if out is None:
            out = np.zeros((num_samples,))
            
        for k in range(num_samples):
            node = 0
            j = self.split_feature[node]
            while j >= 0:
                if self.take_left_node(x[k,j], self.threshold[node], self.left_is_default[node]):
                    node = self.child_left[node]
                else:
                    node = self.child_right[node]
                j = self.split_feature[node]
            out[k] += self.value[node]
            if self.is_linear:
                for j in range(self.num_features):
                    out[k] += self.coeffs[node,j] * x[k,j]
        return np.asarray(out)
    
    
    cdef void shap_mean_baseline(self, double[:,:] x, double[:,:] data, double[:,::1] phi, bint interactions):
        """Computes interventional SHAP as mean of baseline SHAP values (Algorithm 1 in the paper [1]).

        Parameters
        ----------
        x : array of shape (num_samples, num_features)
            Input data for SHAP computation.
        data : array of shape (num_data, num_features)
            Background data for SHAP computation.
        phi : array of shape (num_samples, dim) with dim=num_features or dim=num_features^2
            Array for storing the SHAP (interaction) values. The computed values are added to this array.
            If SHAP values are computed, then `phi` should be of shape (num_samples, num_features).
            If interaction values are computed, then `phi` should be of shape (num_samples, num_features^2), i.e.
            the last dimension stores flattened matrices.
        interactions : bool
            Whether SHAP values (interactions==False) or SHAP interaction values (interactions==True) should be computed.
        """

        cdef:
            SIZE_t num_samples = x.shape[0]
            SIZE_t num_data = data.shape[0]
            SIZE_t k, l, nA, nNB
            NODE_t node, child, child_x, child_data, ancestor
            FEATURE_t j
            bint current_feature_not_in_A, current_feature_not_in_NB
            cpp_vector[NODE_t] stack = cpp_vector[NODE_t]()
            cpp_vector[cnp.int8_t] added_feature_to_A = cpp_vector[cnp.int8_t](self.num_nodes)
            cpp_vector[cnp.int8_t] added_feature_to_NB = cpp_vector[cnp.int8_t](self.num_nodes)
            cpp_vector[FEATURE_t] A = cpp_vector[FEATURE_t]()
            cpp_vector[FEATURE_t] NB = cpp_vector[FEATURE_t]()
            shapley_of_interval update_shap
            double const_coalition
            cpp_vector[double] coeffs_coalition = cpp_vector[double](x.shape[1])
        if self.is_linear:
            if interactions:
                update_shap = shapley_interaction_linear
            else:
                update_shap = shapley_values_linear
        else:
            if interactions:
                update_shap = shapley_interaction_const
            else:
                update_shap = shapley_values_const
        # The following iteration implements Algorithm 1 from the paper. Instead passing (copying) the sets A and B at each split,
        # we keep track of the changes of these sets using the arrays `added_feature_to_A` and `added_feature_to_NB`. If a feature
        # was added to the set A when a node was reached, then added_feature_to_A[node]=1. If the corresponding feature was already
        # in A, then added_feature_to_A[node]=2. Otherwise if the feature is not in A then added_feature_to_A[node]=0. Analogously for 
        # the array `added_feature_to_NB` and the set N\B. When a leaf node is reached, then the sets A and N\B are computed using 
        # these arrays. During the traversal of the tree, these arrays are also used (with the array `self.ancestor_same_feature`) 
        # to check whether a split feature is already in A or in N\B to prevent unnecessary splits.
        A.reserve(self.max_depth)
        NB.reserve(self.max_depth)
        stack.reserve(self.num_nodes)
        for k in range(num_samples):
            for l in range(num_data):
                stack.push_back(0)
                while not stack.empty():
                    node = stack.back()
                    stack.pop_back()
                    j = self.split_feature[node]
                    if j >= 0:
                        if self.take_left_node(x[k,j], self.threshold[node], self.left_is_default[node]):
                            child_x = self.child_left[node]
                        else:
                            child_x = self.child_right[node]
                        if self.take_left_node(data[l,j], self.threshold[node], self.left_is_default[node]):
                            child_data = self.child_left[node]
                        else:
                            child_data = self.child_right[node]
                        ancestor = self.ancestor_same_feature[node]
                        if ancestor >= 0:
                            current_feature_not_in_A = (added_feature_to_A[ancestor] == 0)
                            current_feature_not_in_NB = (added_feature_to_NB[ancestor] == 0)
                        else:
                            current_feature_not_in_A = True
                            current_feature_not_in_NB = True
                        if child_x == child_data:
                            # added_feature_to_A[child_x] = 0 if current_feature_not_in_A else 2
                            if current_feature_not_in_A:
                                added_feature_to_A[child_x] = 0
                            else:
                                added_feature_to_A[child_x] = 2
                            # added_feature_to_NB[child_x] = 0 if current_feature_not_in_NB else 2
                            if current_feature_not_in_NB:
                                added_feature_to_NB[child_x] = 0
                            else:
                                added_feature_to_NB[child_x] = 2
                            stack.push_back(child_x)
                        else:
                            if current_feature_not_in_NB:
                                # added_feature_to_A[child_x] = 1 if current_feature_not_in_A else 2
                                if current_feature_not_in_A:
                                    added_feature_to_A[child_x] = 1
                                else:
                                    added_feature_to_A[child_x] = 2
                                added_feature_to_NB[child_x] = 0
                                stack.push_back(child_x)
                            if current_feature_not_in_A:
                                added_feature_to_A[child_data] = 0
                                # added_feature_to_NB[child_data] = 1 if current_feature_not_in_NB else 2
                                if current_feature_not_in_NB:
                                    added_feature_to_NB[child_data] = 1
                                else:
                                    added_feature_to_NB[child_data] = 2
                                stack.push_back(child_data)
                    else:
                        const_coalition = self.value[node]
                        if self.is_linear:
                            for j in range(self.num_features):
                                const_coalition += self.coeffs[node,j] * data[l,j]
                                coeffs_coalition[j] = self.coeffs[node,j] * (x[k,j] - data[l,j]) / num_data
                        const_coalition /= num_data
                        A.clear()
                        NB.clear()
                        child = node
                        node = self.parent[child]
                        while node >= 0:
                            j = self.split_feature[node]
                            if added_feature_to_A[child] == 1:
                                A.push_back(j)
                            if added_feature_to_NB[child] == 1:
                                NB.push_back(j)
                            child = node
                            node = self.parent[child]
                        update_shap(const_coalition, coeffs_coalition, A, NB, &phi[k,0])
    
    
    cdef void shap_aggregated(self, double[:,:] x, double[:,::1] phi, bint interactions):
        """Computes interventional SHAP using precomputed split statistics (Algorithm 3 in the paper [1]_).

        Parameters
        ----------
        x : array of shape (num_samples, num_features)
            Input data for SHAP computation.
        phi : array of shape (num_samples, dim) with dim=num_features or dim=num_features^2
            Array for storing the SHAP (interaction) values. The computed values are added (in place) to this array.
            If SHAP values are computed, then `phi` should be of shape (num_samples, num_features).
            If interaction values are computed, the `phi` should be of shape (num_samples, num_features^2), i.e.
            the last dimension stores flattened matrices.
        interactions : bool
            Whether SHAP values (interactions==False) or SHAP interaction values (interactions==True) should be computed.
        """

        cdef:
            SIZE_t num_samples = x.shape[0]
            SIZE_t k
            NODE_t node, node0, child, child_lt, child_lf, child_rt, child_rf, ancestor
            FEATURE_t j
            cpp_vector[NODE_t] stack = cpp_vector[NODE_t]()
            cpp_vector[cnp.int8_t] added_feature_to_A = cpp_vector[cnp.int8_t](self.num_nodes_agg)
            cpp_vector[cnp.int8_t] added_feature_to_NB = cpp_vector[cnp.int8_t](self.num_nodes_agg)
            bint current_feature_not_in_A, current_feature_not_in_NB
            cpp_vector[FEATURE_t] A = cpp_vector[FEATURE_t]()
            cpp_vector[FEATURE_t] NB = cpp_vector[FEATURE_t]()
            shapley_of_interval update_shap
            double const_coalition
            cpp_vector[double] coeffs_coalition = cpp_vector[double](x.shape[1])
        if self.is_linear:
            if interactions:
                update_shap = shapley_interaction_linear
            else:
                update_shap = shapley_values_linear
        else:
            if interactions:
                update_shap = shapley_interaction_const
            else:
                update_shap = shapley_values_const
        
        # The following iteration implements Algorithm 3 from the paper. Instead passing (copying) the sets A and B at each split,
        # we keep track of the changes of these sets using the arrays `added_feature_to_A` and `added_feature_to_NB`. If a feature
        # was added to the set A when a node was reached, then added_feature_to_A[node]=1. If the corresponding feature was already
        # in A, then added_feature_to_A[node]=2. Otherwise if the feature is not in A then added_feature_to_A[node]=0. Analogously for 
        # the array `added_feature_to_NB` and the set N\B. When a leaf node is reached, then the sets A and N\B are computed using 
        # these arrays. During the traversal of the tree with aggregated data, these arrays are also used (with the array 
        # `self.ancestor_agg_same_feature`) to check whether a split feature is already in A or in N\B to prevent unnecessary splits.
        A.reserve(self.max_depth)
        NB.reserve(self.max_depth)
        for k in range(num_samples):
            stack.push_back(0)
            while not stack.empty():
                node = stack.back()
                stack.pop_back()
                node0 = self.node_orig[node]
                j = self.split_feature[node0]
                if j >= 0:
                    ancestor = self.ancestor_agg_same_feature[node]
                    if ancestor >= 0:
                        current_feature_not_in_A = (added_feature_to_A[ancestor] == 0)
                        current_feature_not_in_NB = (added_feature_to_NB[ancestor] == 0)
                    else:
                        current_feature_not_in_A = True
                        current_feature_not_in_NB = True
                    child_lt = self.child_left_true[node]
                    child_rt = self.child_right_true[node]
                    if self.take_left_node(x[k,j], self.threshold[node0], self.left_is_default[node0]):
                        if self.cover[child_lt] > 0.:
                            # added_feature_to_A[child_lt] = 0 if current_feature_not_in_A else 2
                            if current_feature_not_in_A:
                                added_feature_to_A[child_lt] = 0
                            else:
                                added_feature_to_A[child_lt] = 2
                            # added_feature_to_NB[child_lt] = 0 if current_feature_not_in_NB else 2
                            if current_feature_not_in_NB:
                                added_feature_to_NB[child_lt] = 0
                            else:
                                added_feature_to_NB[child_lt] = 2
                            stack.push_back(child_lt)
                        if current_feature_not_in_A and self.cover[child_rt] > 0.:
                            added_feature_to_A[child_rt] = 0
                            # added_feature_to_NB[child_rt] = 1 if current_feature_not_in_NB else 2
                            if current_feature_not_in_NB:
                                added_feature_to_NB[child_rt] = 1
                            else:
                                added_feature_to_NB[child_rt] = 2
                            stack.push_back(child_rt)
                        child_lf = self.child_left_false[node]
                        if current_feature_not_in_NB and self.cover[child_lf] > 0.:
                            # added_feature_to_A[child_lf] = 1 if current_feature_not_in_A else 2
                            if current_feature_not_in_A:
                                added_feature_to_A[child_lf] = 1
                            else:
                                added_feature_to_A[child_lf] = 2
                            added_feature_to_NB[child_lf] = 0
                            stack.push_back(child_lf)
                    else:
                        if self.cover[child_rt] > 0.:
                            # added_feature_to_A[child_rt] = 0 if current_feature_not_in_A else 2
                            if current_feature_not_in_A:
                                added_feature_to_A[child_rt] = 0
                            else:
                                added_feature_to_A[child_rt] = 2
                            # added_feature_to_NB[child_rt] = 0 if current_feature_not_in_NB else 2
                            if current_feature_not_in_NB:
                                added_feature_to_NB[child_rt] = 0
                            else:
                                added_feature_to_NB[child_rt] = 2
                            stack.push_back(child_rt)
                        if current_feature_not_in_A and self.cover[child_lt] > 0.:
                            added_feature_to_A[child_lt] = 0
                            # added_feature_to_NB[child_lt] = 1 if current_feature_not_in_NB else 2
                            if current_feature_not_in_NB:
                                added_feature_to_NB[child_lt] = 1
                            else:
                                added_feature_to_NB[child_lt] = 2
                            stack.push_back(child_lt)
                        child_rf = self.child_right_false[node]
                        if current_feature_not_in_NB and self.cover[child_rf] > 0.:
                            # added_feature_to_A[child_rf] = 1 if current_feature_not_in_A else 2
                            if current_feature_not_in_A:
                                added_feature_to_A[child_rf] = 1
                            else:
                                added_feature_to_A[child_rf] = 2
                            added_feature_to_NB[child_rf] = 0
                            stack.push_back(child_rf)
                else:
                    const_coalition = self.value[node0] * self.cover[node]
                    if self.is_linear:
                        for j in range(self.num_features):
                            const_coalition += self.coeffs[node0,j] * self.mean_data[node,j] * self.cover[node]
                            coeffs_coalition[j] = self.coeffs[node0,j] * (x[k,j] - self.mean_data[node,j]) \
                                                  * self.cover[node]
                    A.clear()
                    NB.clear()
                    child = node
                    node = self.parent_agg[child]
                    while node >= 0:
                        node0 = self.node_orig[node]
                        j = self.split_feature[node0]
                        if added_feature_to_A[child] == 1:
                            A.push_back(j)
                        if added_feature_to_NB[child] == 1:
                            NB.push_back(j)
                        child = node
                        node = self.parent_agg[child]
                    update_shap(const_coalition, coeffs_coalition, A, NB, &phi[k,0]) 
                        
    
    def shap_values(self, double[:,:] x, double[:,:] data=None, double[:,::1] phi=None):
        """Computes interventional SHAP values for inputs `x`.

        If background data (`data`) is provided, the SHAP values are computed by iterating over 
        background data points (Algorithm 1 in the paper [1]_). Otherwise precomputed split statistics
        are used for SHAP computation (Algorithm 3 in the paper [1]_).

        Parameters
        ----------
        x : array of shape (num_samples, num_features)
            Input data for SHAP computation.
        data : None or array of shape (num_data, num_features)
            Background data for SHAP computation. Either the tree should hold aggregated data or background data should be provided.
        phi : None or array of shape (num_samples, num_features)
            Array for the output. If provided, the computed values are added (in place) to this array (used for SHAP values of ensembles).

        Returns
        -------
        phi : array of shape (num_samples, num_features)
            Array holding the computed SHAP values.
        """

        if self.is_linear:
            assert x.shape[1] == self.num_features
        else:
            assert x.shape[1] >= self.num_features
        if phi is None:
            phi = np.zeros_like(x, dtype=np.float64)
        if data is None:
            if self.has_aggregated:
                self.shap_aggregated(x, phi, False)
            else:
                raise ValueError("Either `data` should be given or the tree should hold aggregated data!")
        else:
            self.shap_mean_baseline(x, data, phi, False)
        return np.asarray(phi, dtype=np.float64)
    
    
    def shap_interaction_values(self, double[:,:] x, double[:,:] data=None, double[:,:,::1] phi=None):
        """Computes interventional SHAP interaction values for inputs `x`.

        If background data (`data`) is provided, the SHAP interaction values are computed by iterating over 
        background data points (Algorithm 1 in the paper [1]_). Otherwise precomputed split statistics
        are used for SHAP computation (Algorithm 3 in the paper [1]_).
        The output array holds matrices phi[k,:,:] for each sample point x[k,:]. The nondiagonal elements
        are the corresponding interaction values scaled by 0.5, i.e. phi[k,i,j] = 0.5 * phi_{i,j}(x[k,:]).
        The diagonal elements store the corresponding SHAP values corrected by the interaction values, i.e.
        phi[k,i,i] = phi_{i}(x[k,:]) - 0.5 * sum_{j!=i} phi_{i,j}(x[k,:]).

        Parameters
        ----------
        x : array of shape (num_samples, num_features)
            Input data for SHAP computation.
        data : None or array of shape (num_data, num_features)
            Background data for SHAP computation. Either the tree should hold aggregated data or background data should be provided.
        phi : None or array of shape (num_samples, num_features, num_features)
            Array for the output. If provided, the computed values are added (in place) to this array (used for SHAP of ensembles).

        Returns
        -------
        phi : array of shape (num_samples, num_features, num_features)
            Array holding the computed SHAP interaction values.
        """

        cdef double[:,::1] phi_reshaped
        if self.is_linear:
            assert x.shape[1] == self.num_features
        else:
            assert x.shape[1] >= self.num_features
        if phi is None:
            phi = np.zeros((x.shape[0], x.shape[1], x.shape[1]), dtype=np.float64)
        phi_reshaped = np.reshape(phi, (x.shape[0], x.shape[1] * x.shape[1]))
        if data is None:
            if self.has_aggregated:
                self.shap_aggregated(x, phi_reshaped, True)
            else:
                raise ValueError("Either `data` should be given or the tree should hold aggregated data!")
        else:
            self.shap_mean_baseline(x, data, phi_reshaped, True)
        return np.reshape(phi_reshaped, (x.shape[0], x.shape[1], x.shape[1]))
