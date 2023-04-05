from typing import List
from ncpol2sdpa import*
import numpy as np
from sklearn.preprocessing import scale
from itertools import combinations
from sklearn.gaussian_process import GaussianProcessRegressor
from castle.common import BaseLearner, Tensor
from castle.common.independence_tests import hsic_test

"""
variable example

Y=[1,2,3]
X=[1,2,3]
T=3
level=1

"""

def ncpop(Y,X,level=1):

    # T=Y.shape[0]
    T = len(Y)

    # Decision Variables
    G = generate_operators("G", n_vars=1, hermitian=True, commutative=True)[0]
    f = generate_operators("f", n_vars=T, hermitian=True, commutative=True)
    n = generate_operators("m", n_vars=T, hermitian=True, commutative=True)

    # Objective
    obj = sum((Y[i]-f[i])**2 for i in range(T)) + 0.5*sum(f[i]**2 for i in range(T))
    # Constraints
    ine1 = [f[i] - G*X[i] - n[i] for i in range(T)]
    ine2 = [-f[i] + G*X[i] + n[i] for i in range(T)]
    ines = ine1+ine2

    # Solve the NCPO
    sdp = SdpRelaxation(variables = flatten([G,f,n]),verbose = 1)
    sdp.get_relaxation(level, objective=obj, inequalities=ines)
    sdp.solve(solver='mosek')
    #sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:/Users/zhouq/Documents/sdpa7-windows/sdpa.exe"})
    print(sdp.primal, sdp.dual, sdp.status)

    if(sdp.status != 'infeasible'):
        print('ok.')
        est_noise = []
        for i in range(T):
            est_noise.append(sdp[n[i]])
        return est_noise
    else:
        print('Cannot find feasible solution.')
        return



class NCPLinear(BaseLearner):
    """
    # introductions

    References
    ----------
    Quan Zhou

    Parameters
    ----------
    alpha : float, default 0.05
        significance level be used to compute threshold

    Attributes
    ----------
    causal_matrix : array like shape of (n_features, n_features)
        Learned causal structure matrix.

    Examples
    --------
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> from castle.datasets import DAG, IIDSimulation
    >>> from castle.algorithms.anm import ANMNonlinear

    >>> weighted_random_dag = DAG.erdos_renyi(n_nodes=6, n_edges=10,
    >>>                                      weight_range=(0.5, 2.0), seed=1)
    >>> dataset = IIDSimulation(W=weighted_random_dag, n=1000,
    >>>                         method='nonlinear', sem_type='gp-add')
    >>> true_dag, X = dataset.B, dataset.X

    >>> anm = ANMNonlinear(alpha=0.05)
    >>> anm.learn(data=X)

    >>> # plot predict_dag and true_dag
    >>> GraphDAG(anm.causal_matrix, true_dag, show=False, save_name='result')

    """

    def __init__(self,alpha=0.05):
        super(NCPLinear, self).__init__()
        self.alpha = alpha

    def learn(self, data, columns=None, test_method=hsic_test, **kwargs):
        """Set up and run the ncpop algorithm.

        Parameters
        ----------
        data: numpy.ndarray or Tensor
            Training data.
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        regressor: Class
            Nonlinear regression estimator, if not provided, it is GPR.
            If user defined, must implement `estimate` method. such as :
                `regressor.estimate(x, y)`
        test_method: callable, default test_method
            independence test method, if not provided, it is HSIC.
            If user defined, must accept three arguments--x, y and keyword
            argument--alpha. such as :
                `test_method(x, y, alpha=0.05)`
        """

        # self.regressor = regressor

        # create learning model and ground truth model
        data = Tensor(data, columns=columns)

        node_num = data.shape[1]
        self.causal_matrix = Tensor(np.zeros((node_num, node_num)),
                                    index=data.columns,
                                    columns=data.columns)

        for i, j in combinations(range(node_num), 2):
            x = data[:, i].reshape((-1, 1))
            y = data[:, j].reshape((-1, 1))

            flag = test_method(x, y, alpha=self.alpha)
            if flag == 1:
                continue
            # test x-->y
            flag = self.ncpop_estimate(x, y, 
                                     test_method=test_method)
            if flag:
                self.causal_matrix[i, j] = 1
            # test y-->x
            flag = self.ncpop_estimate(y, x, 
                                     test_method=test_method)
            if flag:
                self.causal_matrix[j, i] = 1

    def ncpop_estimate(self, x, y, test_method=hsic_test):
        """Compute the fitness score of the ncpop model in the x->y direction.

        Parameters
        ----------
        x: array
            Variable seen as cause
        y: array
            Variable seen as effect
        regressor: Class
            Nonlinear regression estimator, if not provided, it is GPR.
            If user defined, must implement `estimate` method. such as :
                `regressor.estimate(x, y)`
        test_method: callable, default test_method
            independence test method, if not provided, it is HSIC.
            If user defined, must accept three arguments--x, y and keyword
            argument--alpha. such as :
                `test_method(x, y, alpha=0.05)`

        Returns
        -------
        out: int, 0 or 1
            If 1, residuals n is independent of x, then accept x --> y
            If 0, residuals n is not independent of x, then reject x --> y

        Examples
        --------
        >>> import numpy as np
        >>> from castle.algorithms.anm import ANMNonlinear
        >>> np.random.seed(1)
        >>> x = np.random.rand(500, 2)
        >>> anm = ANMNonlinear(alpha=0.05)
        >>> print(anm.anm_estimate(x[:, [0]], x[:, [1]]))
        1
        """

        x = scale(x).reshape((-1, 1))
        y = scale(y).reshape((-1, 1))

        X = list(x.flatten())
        Y = list(y.flatten())

        # y_predict = ncpop(Y,X)
        est_noise = ncpop(Y,X)
        flag = test_method(np.array(est_noise).reshape((-1,1)), x, alpha=self.alpha)

        return flag