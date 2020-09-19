from __future__ import print_function
import numpy as np
import scipy.stats as stats

def bic_from_likelihood(lnL,k,n):
    """
    Baysian information criterion (BIC). The model with the lowest BIC is preferred.
    
    INPUT:
        lnL - maximum likelihood value
        k - number of model parameters in the test
        n - number of points
        
    NOTE:
        lower BIC is better
        https://en.wikipedia.org/wiki/Bayesian_information_criterion

        DeltaBIC       Evidence against higher BIC
        0 to 2	       Not worth more than a bare mention
        2 to 6	       Positive
        6 to 10	       Strong
        >10	       Very strong
    """
    bic = -2.*lnL + k*np.log(n)
    return bic

def bic(residuals,errors,k,verbose=False):
    """
    Baysian information criterion (BIC). The model with the lowest BIC is preferred.
    
    INPUT:
        residuals - residuals 
        k - number of model parameters in the test
        
    NOTE:
        CAN ONLY DO MODEL COMPARISON IF THE ERRORS ARE THE SAME (Gibson et al. 2014)

        lower BIC is better
        https://en.wikipedia.org/wiki/Bayesian_information_criterion

        DeltaBIC       Evidence against higher BIC
        0 to 2	       Not worth more than a bare mention
        2 to 6	       Positive
        6 to 10	       Strong
        >10	       Very strong
    """
    _chi2 = chi2(residuals,errors,k,verbose=verbose)
    #_chi2 = np.sum(residuals**2.)
    n = len(residuals)
    #bic = n*np.log(_chi2/n) + k*np.log(n)
    bic = _chi2 + k*np.log(n)
    return bic

def chi2(residuals,errors,k,return_reduced=False,verbose=True):
    """
    Weighted chi2, weighted by errors.

    INPUT:
        residuals - an array of residuals
        errors - corresponding errors
        k - the number of model parameters
    """
    chi2 = np.sum((residuals/errors)**2.)
    n = len(residuals)
    dof = n-k
    p_value = 1.-stats.chi2.cdf(x=chi2,df=dof)
    if verbose:
        print('Chi2: {}'.format(chi2))
        print('Chi2 reduced: {}'.format(chi2/dof))
        print('DOF: {}'.format(dof))
        print('Assuming the model is correct, there is {:0.3f}% chance\nthat this chi2 value or larger could arise by chance'.format(p_value*100))
        print('If p value is low, model is ruled out. If p value is high, supports model')
    if return_reduced:
        return chi2/dof
    else:
        return chi2

def aic(k,lnL):
    """
    Akaike Information Criterion -- the lowest value is preferred.

    INPUT:
        k - number of parameters in the model
        lnL - log likelihood

    NOTES:
        The relative likelihood is definied as exp(Delta AIC/2) Grunblatt et al. 2015

    https://en.wikipedia.org/wiki/Akaike_information_criterion
    """
    return 2.*(k-lnL)

#sm.distributions.ECDF()
#import statsmodels as sm
#sample = df_mcmc.k1.values
#ecdf = sm.distributions.ECDF(sample)
#x = np.linspace(min(sample), max(sample),2000)
#y = ecdf(x)
#plt.step(x, y)
#scipy.interpolate.interp1d(y,x)(0.997)
#np.quantile(df_mcmc.k1.values,0.997)
