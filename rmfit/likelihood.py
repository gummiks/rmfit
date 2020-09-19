from numpy import square, log, ndarray, pi
from math import log as mlog
#from .likelihood_f import lh as lhf
LOG_TWO_PI = log(2*pi)

def ll_normal_es_py(o,m,e):
    """Normal log likelihood for scalar average standard deviation."""
    npt = o.size
    return -npt*mlog(e) -0.5*npt*LOG_TWO_PI - 0.5*square(o-m).sum()/e**2

def ll_normal_ev_py(o,m,e):
    """Normal log likelihood for varying e"""
    npt = o.size
    return -log(e).sum() -0.5*npt*LOG_TWO_PI - 0.5*square((o-m)/e).sum()

