# this file contains collections of proxes we learned in the class
import numpy as np
from scipy.optimize import bisect

# =============================================================================
# TODO Complete the following prox for simplex
# =============================================================================

# Prox of capped simplex
# -----------------------------------------------------------------------------


def prox_csimplex(z, k):
    """
    Prox of capped simplex
            argmin_x 1/2||x - z||^2 s.t. x in k-capped-simplex.

    input
    -----
    z : arraylike
            reference point
    k : int
            positive number between 0 and z.size, denote simplex cap

    output
    ------
    x : arraylike
            projection of z onto the k-capped simplex
    """
    # safe guard for k
    assert 0 <= k <= z.size, 'k: k must be between 0 and dimension of the input.'

    # TODO do the computation here
    # Hint: 1. construct the scalar dual object and use `bisect` to solve it.
    # 2. obtain primal variable from optimal dual solution and return it.
    #
    def DualFunc(lam):
        df = np.sum(np.maximum(0, np.minimum(1, z - lam))) - k
        return df
    
    def xMin(z, lam): 
        xm = np.maximum(0, np.minimum(1, z - lam))
        return xm
    
    a = -np.sum(np.abs(z))
    b = np.sum(np.abs(z))
    
    minLam = bisect(DualFunc, a , b)
    xmin = xMin(z, minLam)
    return xmin




def prox_l1(x, t):
    """
    regular l1 prox included for convenience
    Note that you'll have to rescale the t input with the regularization parameter
    """
    y = np.zeros(x.size)
    ind = np.where(np.abs(x) > t)
    x_o = x[ind]
    y[ind] = np.sign(x_o)*(np.abs(x_o) - t)
    return y


def rank_project(Y, k):
    """	Prox of rank constrained matrices
            argmin_M 1/2||M - Y||^2 s.t. rank(M)<=k


    Parameters
    ----------
    Y : 2 dimensional array
    k : positive integer

    Returns
    -------
    2 dimensional array
            rank projected version of Y
    """
    U,sigma,Vt=np.linalg.svd(Y)
    sigma[k:]=0
    X=(U@np.diag(sigma)@Vt)
    return X


def nuclear_prox(Y, t):
    """Nuclear norm proximal operator
    argmin_M 1/(2t)||M - Y||^2 + ||M||_{*}

    Parameters
    ----------
    Y : 2 dimensional array
    k : positive integer

    Returns
    -------
    2 dimensional array
            proximal operator applied to Y
    """
    # Do SVD of Y
    # Do the prox of 1 norm on sigma
    # diag(prox_1norm of sigma)
    # plug back into to SVD
    U,sigma,Vt = np.linalg.svd(Y,full_matrices=False)
    prox_sigma = np.sign(sigma)*np.maximum(np.abs(sigma) - t, 0)
#     X = (U[:,:5]@np.diag(prox_sigma[:5])@Vt[:5,:])
    X = (U@np.diag(prox_sigma)@Vt)
    return X
    
  
