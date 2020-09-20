import numpy as np
import pandas as pd
import corner
import matplotlib.pyplot as plt
from matplotlib import rcParams

def plot_chains(chain,labels=None,burn=0,thin=1,plot_priors=False,alpha=0.1):
    """
    Plot MCMC chains

    chain = self.sampler.chain[:,burn::thin,:]
    """
    chain = chain[:,burn::thin,:]
    ndim = chain.shape[2]
    nwalkers = chain.shape[0]
    fig, ax = plt.subplots(nrows=ndim,sharex=True,dpi=100)
    for i in range(ndim):
        for walker in range(nwalkers):
            ax[i].plot(chain[walker,:,i],color="black",alpha=alpha,lw=0.5);
        if labels:
            ax[i].set_ylabel(labels[i],fontsize=8)
        ax[i].margins(y=0.1)
        for label in ax[i].get_yticklabels():
            label.set_fontsize(6)

    ax[i].set_xlabel("sample",fontsize=8)
    ax[i].minorticks_on()
    ax[0].set_title("Overview of chains",y=1.03,fontsize=12)
    for label in ax[i].get_xticklabels():
            label.set_fontsize(6)
    fig.subplots_adjust(hspace=0.015)

def plot_corner(chain,labels=None,burn=0,thin=1,title_fmt=".5f",xlabcord=(0.5,-0.2),ylabcord=(-0.2,0.5),**kwargs):
    """
    Plot a corner plot of the jump parameters.
    """
    rcParams["lines.linewidth"] = 1.0
    rcParams["axes.labelpad"] = 20.0
    rcParams["xtick.labelsize"] = 14.0
    rcParams["ytick.labelsize"] = 14.0

    ndim = chain.shape[2]
    chain = chain[:,burn::thin,:].reshape((-1, ndim))

    figure = corner.corner(chain,
                           labels=labels,
                           quantiles=[0.16, 0.5, 0.84],
                           verbose=False,
                           title_kwargs={"fontsize": 14},
                           hist_kwargs={"lw":1.},
                           label_kwargs={"fontsize":18},
                           xlabcord=xlabcord,
                           ylabcord=ylabcord,
                           title_fmt=title_fmt,**kwargs)
    return figure

def gelman_rubin(chains):
    """
    Calculates the gelman rubin statistic.

    # NOTE: 
        Should be close to 1
    """
    nwalker = chains.shape[0]
    niter = chains.shape[1]
    npar = chains.shape[2]
    grarray = np.zeros(npar)
    for i in range(npar):
        sj2 = np.zeros(nwalker)
        chainmeans = np.zeros(nwalker)
        for j in range(nwalker):
            chainmeans[j] = np.mean(chains[j,:,i])
            sj2[j] = np.sum((chains[j,:,i]-chainmeans[j])**2.) / (niter-1)
        W = np.sum(sj2) / nwalker
        ThetaDoubleBar = np.sum(chainmeans) / nwalker
        B = np.sum((chainmeans-ThetaDoubleBar)**2.) * niter / (nwalker-1)
        VarTheta = (1-(1/niter))*W + (B/niter)
        grarray[i] = np.sqrt(VarTheta/W)
    return grarray
