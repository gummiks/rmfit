import numpy as np
import pandas as pd
import pyde
import pyde.de
import matplotlib.pyplot as plt
import emcee
import batman
import astropy.constants as aconst
import radvel
from .priors import PriorSet, UP, NP, JP, FP
from .likelihood import ll_normal_ev_py
from . import stats_help
from . import utils
from . import mcmc_help

class LPFunction(object):
    """
    Log-Likelihood function class

    NOTES:
        Based on hpprvi's class, see: https://github.com/hpparvi/exo_tutorials
    """
    def __init__(self,x,y,yerr,file_priors):
        """
        INPUT:
            x - time values in BJD
            y - y values in m/s
            yerr - yerr values in m/s
            file_priors - prior file name
        """
        self.data= {"x"   : x,
                    "y"   : y,
                    "error"  : yerr}
        # Setting priors
        self.ps_all = priorset_from_file(file_priors) # all priors
        self.ps_fixed = PriorSet(np.array(self.ps_all.priors)[np.array(self.ps_all.fixed)]) # fixed priorset
        self.ps_vary  = PriorSet(np.array(self.ps_all.priors)[~np.array(self.ps_all.fixed)]) # varying priorset
        self.ps_fixed_dict = {key: val for key, val in zip(self.ps_fixed.labels,self.ps_fixed.args1)}
        print('Reading in priorfile from {}'.format(file_priors))
        print(self.ps_all.df)

    def get_jump_parameter_index(self,lab):
        """
        Get the index of a given label
        """
        return np.where(np.array(self.ps_vary.labels)==lab)[0][0]

    def get_jump_parameter_value(self,pv,lab):
        """
        Get the current value in the argument list 'pv' that has label 'lab'
        """
        # First check if we are actually varying it
        if lab in self.ps_vary.labels:
            return pv[self.get_jump_parameter_index(lab)]
        else:
            # We are not varying it
            return self.ps_fixed_dict[lab]

    def compute_rm_model(self,pv,times=None):
        """
        Calls RM model and returns the transit model

        INPUT:
            pv    - parameters passed to the function
            times - times, and array of timestamps

        OUTPUT:
            lc - the lightcurve model at *times*
        """
        T0     =self.get_jump_parameter_value(pv,'t0_p1')
        P      =self.get_jump_parameter_value(pv,'P_p1')
        lam    =self.get_jump_parameter_value(pv,'lam_p1')
        vsini  =self.get_jump_parameter_value(pv,'vsini')
        ii     =self.get_jump_parameter_value(pv,'inc_p1')
        rprs   =self.get_jump_parameter_value(pv,'p_p1')
        aRs    =self.get_jump_parameter_value(pv,'a_p1')
        u1     =self.get_jump_parameter_value(pv,'u1')
        u2     =self.get_jump_parameter_value(pv,'u2')
        gamma  =self.get_jump_parameter_value(pv,'gamma')
        beta   =self.get_jump_parameter_value(pv,'vbeta')
        #sigma  =self.get_jump_parameter_value(pv,'sigma')
        sigma  = vsini /1.31 # assume sigma is vsini/1.31 (see Hirano et al. 2010, 2011)
        e      =self.get_jump_parameter_value(pv,'ecc_p1')
        omega  =self.get_jump_parameter_value(pv,'omega_p1')
        exptime=self.get_jump_parameter_value(pv,'exptime')/86400. # exptime in days
        if times is None:
            times = self.data["x"]
        self.RH = RMHirano(lam,vsini,P,T0,aRs,ii,rprs,e,omega,[u1,u2],beta,
                            sigma,supersample_factor=7,exp_time=exptime,limb_dark='quadratic')
        self.rm = self.RH.evaluate(times)
        return self.rm

    def compute_rv_model(self,pv,times=None):
        """
        Compute the RV model

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rv - the rv model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps
        """
        if times is None:
            times = self.data["x"]
        T0      = self.get_jump_parameter_value(pv,'t0_p1')
        P       = self.get_jump_parameter_value(pv,'P_p1')
        gamma   = self.get_jump_parameter_value(pv,'gamma')
        K       = self.get_jump_parameter_value(pv,'K_p1')
        e       = self.get_jump_parameter_value(pv,'ecc_p1')
        w       = self.get_jump_parameter_value(pv,'omega_p1')
        self.rv = get_rv_curve(times,P=P,tc=T0,e=e,omega=w,K=K)+gamma
        return self.rv

    def compute_total_model(self,pv,times=None):
        """
        Computes the full RM model (including RM and RV)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps

        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model()
        """
        return self.compute_rm_model(pv,times=times) + self.compute_rv_model(pv,times=times)

    def __call__(self,pv):
        """
        Return the log likelihood

        INPUT:
            pv - the input list of varying parameters
        """
        if any(pv < self.ps_vary.pmins) or any(pv>self.ps_vary.pmaxs):
            return -np.inf

        ###############
        # Prepare data and model and error for ingestion into likelihood
        y_data = self.data['y']
        y_model = self.compute_total_model(pv)
        # jitter in quadrature
        jitter = self.get_jump_parameter_value(pv,'sigma_rv')
        error = np.sqrt(self.data['error']**2.+jitter**2.)
        ###############

        # Return the log-likelihood
        log_of_priors = self.ps_vary.c_log_prior(pv)
        # Calculate log likelihood
        log_of_model  = ll_normal_ev_py(y_data, y_model, error)
        log_ln = log_of_priors + log_of_model
        return log_ln

class RMFit(object):
    """
    A class that does RM fitting.

    NOTES:
        - Needs to have LPFunction defined
    """
    def __init__(self,LPFunction):
        self.lpf = LPFunction

    def minimize_AMOEBA(self):
        centers = np.array(self.lpf.ps_vary.centers)

        def neg_lpf(pv):
            return -1.*self.lpf(pv)
        self.min_pv = minimize(neg_lpf,centers,method='Nelder-Mead',tol=1e-9,
                                   options={'maxiter': 100000, 'maxfev': 10000, 'disp': True}).x

    def minimize_PyDE(self,npop=100,de_iter=200,mc_iter=1000,mcmc=True,threads=8,maximize=True,plot_priors=True,sample_ball=False,k=None,n=None):
        """
        Minimize using the PyDE

        NOTES:
            see https://github.com/hpparvi/PyDE
        """
        centers = np.array(self.lpf.ps_vary.centers)
        print("Running PyDE Optimizer")
        self.de = pyde.de.DiffEvol(self.lpf, self.lpf.ps_vary.bounds, npop, maximize=maximize) # we want to maximize the likelihood
        self.min_pv, self.min_pv_lnval = self.de.optimize(ngen=de_iter)
        print("Optimized using PyDE")
        print("Final parameters:")
        self.print_param_diagnostics(self.min_pv)
        #self.lpf.ps.plot_all(figsize=(6,4),pv=self.min_pv)
        print("LogPost value:",-1*self.min_pv_lnval)
        self.lnl_max  = -1*self.min_pv_lnval-self.lpf.ps_vary.c_log_prior(self.min_pv)
        print("LnL value:",self.lnl_max)
        print("Log priors",self.lpf.ps_vary.c_log_prior(self.min_pv))
        if k is not None and n is not None:
            print("BIC:",stats_help.bic_from_likelihood(self.lnl_max,k,n))
            print("AIC:",stats_help.aic(k,self.lnl_max))
        if mcmc:
            print("Running MCMC")
            self.sampler = emcee.EnsembleSampler(npop, self.lpf.ps_vary.ndim, self.lpf,threads=threads)

            #pb = ipywidgets.IntProgress(max=mc_iter/50)
            #display(pb)
            #val = 0
            print("MCMC iterations=",mc_iter)
            for i,c in enumerate(self.sampler.sample(self.de.population,iterations=mc_iter)):
                print(i,end=" ")
                #if i%50 == 0:
                    #val+=50.
                    #pb.value += 1
            print("Finished MCMC")
            self.min_pv_mcmc = self.get_mean_values_mcmc_posteriors().medvals.values

    def get_mean_values_mcmc_posteriors(self,flatchain=None):
        """
        Get the mean values from the posteriors

            flatchain - if not passed, then will default using the full flatchain (will likely include burnin)

        EXAMPLE:
        """
        if flatchain is None:
            flatchain = self.sampler.flatchain
            print('No flatchain passed, defaulting to using full chains')
        df_list = [utils.get_mean_values_for_posterior(flatchain[:,i],label,description) for i,label,description in zip(range(len(self.lpf.ps_vary.descriptions)),self.lpf.ps_vary.labels,self.lpf.ps_vary.descriptions)]
        return pd.concat(df_list)

    def print_param_diagnostics(self,pv):
        """
        A function to print nice parameter diagnostics.
        """
        self.df_diagnostics = pd.DataFrame(zip(self.lpf.ps_vary.labels,self.lpf.ps_vary.centers,self.lpf.ps_vary.bounds[:,0],self.lpf.ps_vary.bounds[:,1],pv,self.lpf.ps_vary.centers-pv),columns=["labels","centers","lower","upper","pv","center_dist"])
        print(self.df_diagnostics.to_string())
        return self.df_diagnostics

    def plot_fit(self,pv=None,times=None):
        """
        Plot the model curve for a given set of parameters pv

        INPUT:
            pv - an array containing a sample draw of the parameters defined in self.lpf.ps_vary
               - will default to best-fit parameters if none are supplied
        """
        if pv is None:
            print('Plotting curve with best-fit values')
            pv = self.min_pv
        x = self.lpf.data['x']
        y = self.lpf.data['y']
        jitter = self.lpf.get_jump_parameter_value(pv,'sigma_rv')
        yerr = np.sqrt(self.lpf.data['error']**2.+jitter**2.)
        model = self.lpf.compute_total_model(pv)
        residuals = y-model

        # Plot
        nrows = 2
        self.fig, self.ax = plt.subplots(nrows=nrows,sharex=True,figsize=(10,6),gridspec_kw={'height_ratios': [5, 2]})
        self.ax[0].errorbar(x,y,yerr=yerr,elinewidth=1,lw=0,alpha=1,capsize=5,mew=1,marker="o",barsabove=True,markersize=8,label="Data")
        self.ax[0].plot(x,model,label="Model",color='crimson')
        self.ax[1].errorbar(x,residuals,yerr=yerr,elinewidth=1,lw=0,alpha=1,capsize=5,mew=1,marker="o",barsabove=True,markersize=8,
                            label="Residuals, std="+str(np.std(residuals)))
        for xx in self.ax:
            xx.minorticks_on()
            xx.legend(loc='lower left',fontsize=9)
            xx.set_ylabel("RV [m/s]",labelpad=2)
        self.ax[-1].set_xlabel("Time (BJD)",labelpad=2)
        self.ax[0].set_title("RM Effect")
        self.fig.subplots_adjust(wspace=0.05,hspace=0.05)

    def plot_mcmc_fit(self,times=None):
        df = self.get_mean_values_mcmc_posteriors()
        print('Plotting curve with best-fit mcmc values')
        self.plot_fit(pv=df.medvals.values)

def read_priors(priorname):
    """
    Read a prior file as in juliet.py style

    OUTPUT:
        priors - prior dictionary
        n_params - number of parameters

    EXAMPLE:
        P, numpriors = read_priors('../data/priors.dat')
    """
    fin = open(priorname)
    priors = {}
    n_transit = 0
    n_rv = 0
    n_params = 0
    numbering_transit = np.array([])
    numbering_rv = np.array([])
    while True:
        line = fin.readline()
        if line != '':
            if line[0] != '#':
                line = line.split('#')[0] # remove things after comment
                out = line.split()
                parameter,prior_name,vals = out
                parameter = parameter.split()[0]
                prior_name = prior_name.split()[0]
                vals = vals.split()[0]
                priors[parameter] = {}
                pvector = parameter.split('_')
                # Check if parameter/planet is from a transiting planet:
                if pvector[0] == 'r1' or pvector[0] == 'p':
                    pnumber = int(pvector[1][1:])
                    numbering_transit = np.append(numbering_transit,pnumber)
                    n_transit += 1
                # Check if parameter/planet is from a RV planet:
                if pvector[0] == 'K':
                    pnumber = int(pvector[1][1:])
                    numbering_rv = np.append(numbering_rv,pnumber)
                    n_rv += 1
                if prior_name.lower() == 'fixed':
                    priors[parameter]['type'] = prior_name.lower()
                    priors[parameter]['value'] = np.double(vals)
                    priors[parameter]['cvalue'] = np.double(vals)
                else:
                    n_params += 1
                    priors[parameter]['type'] = prior_name.lower()
                    if priors[parameter]['type'] != 'truncatednormal':
                        v1,v2 = vals.split(',')
                        priors[parameter]['value'] = [np.double(v1),np.double(v2)]
                    else:
                        v1,v2,v3,v4 = vals.split(',')
                        priors[parameter]['value'] = [np.double(v1),np.double(v2),np.double(v3),np.double(v4)]
                    priors[parameter]['cvalue'] = 0.
        else:
            break
    #return priors, n_transit, n_rv, numbering_transit.astype('int'), numbering_rv.astype('int'), n_params
    return priors, n_params

def priordict_to_priorset(priordict,verbose=True):
    """
    Get a PriorSet from prior diectionary

    EXAMPLE:
        P, numpriors = readpriors('../data/priors.dat')
        ps = priordict_to_priorset(priors)
        ps.df
    """
    priors = []
    for key in priordict.keys():
        inp = priordict[key]
        if verbose: print(key)
        val = inp['value']
        if inp['type'] == 'normal':
            outp = NP(val[0],val[1],key,key,priortype='model')
        elif inp['type'] == 'uniform':
            outp = UP(val[0],val[1],key,key,priortype='model')
        elif inp['type'] == 'fixed':
            outp = FP(val,key,key,priortype='model')
        else:
            print('Error, ptype {} not supported'.format(inp['type']))
        priors.append(outp)
    return PriorSet(priors)

def priorset_from_file(filename,verbose=False):
    """
    Get a PriorSet() from a filename
    """
    priordict, num_priors = read_priors(filename)
    return priordict_to_priorset(priordict,verbose)

class RMHirano(object):
    def __init__(self,lam,vsini,P,T0,aRs,i,RpRs,e,w,u,beta,sigma,supersample_factor=7,exp_time=0.00035,limb_dark='linear'):
        """
        Evaluate Rossiter McLaughlin effect using the model of Hirano et al. 2010, 2011

        INPUT:
            lam - sky-projected obliquity in deg
            vsini - sky-projected rotational velocity in km/s
            P - Period in days
            T0 - Transit center
            aRs - a/R*
            i - inclination in degrees
            RpRs - radius ratio
            e - eccentricity
            w - omega in degrees
            u - [u1,u2] where u1 and u2 are the quadratic limb-dark coefficients
            beta - Gaussian dispersion of spectral lines, in km/s, typically 2.5-4.5km/s (see Hirano+11)
            sigma - Gaussian broadening kernel from Hirano+10. Hirano+10 found vsini/1.31 as an approximation that sometimes works (Why?).

        EXAMPLE:
            times = np.linspace(-0.05,0.05,200)
            T0 = 0.
            P = 3.48456408
            aRs = 21.09
            i = 89.
            vsini = 8.7
            rprs = np.sqrt(0.01)
            e = 0.
            w = 90.
            lam = 45.
            u = [0.3,0.3]
            R = RMHirano(lam,vsini,P,T0,aRs,i,rprs,e,w,u,limb_dark='quadratic')
            rm = R.evaluate(times)

            fig, ax = plt.subplots()
            ax.plot(times,rm)
        """
        self.lam = lam
        self.vsini = vsini
        self.P = P
        self.T0 = T0
        self.aRs = aRs
        self.i = i
        self.iS = 90. # not really using anymore, as it cancels out, should just remove
        self.RpRs = RpRs
        self.e = e
        self.w = w
        self.u = u
        self.limb_dark = limb_dark
        self.rstar = 1. # not really using anymore, as it cancels out, should just remove
        self.beta = beta
        self.sigma = sigma
        self.exp_time = exp_time
        self.supersample_factor = int(supersample_factor)
        self._Omega = (self.vsini/np.sin(np.deg2rad(self.iS)))/(self.rstar*aconst.R_sun.value/1000.)

    def true_anomaly(self,times):
        """
        Calculate the true anomaly
        """
        f = true_anomaly(times,self.T0,self.P,self.aRs,self.i,self.e,self.w)
        return f

    def calc_transit(self,times):
        """
        Calculate transit model of planet
        """
        params = batman.TransitParams()
        params.t0 = self.T0
        params.per = self.P
        params.inc = self.i
        params.rp = self.RpRs
        params.a = self.aRs
        params.ecc = self.e
        params.w = self.w
        params.u = self.u
        params.limb_dark = self.limb_dark
        params.fp = 0.001
        transitmodel = batman.TransitModel(params, times, transittype='primary',exp_time=self.exp_time,
                                         supersample_factor=self.supersample_factor)
        return transitmodel.light_curve(params)

    def planet_XYZ_position(self,times):
        """
        Get the planet XYZ position at times
        """
        X, Y, Z = planet_XYZ_position(times,self.T0,self.P,self.aRs,self.i,self.e,self.w)
        return X, Y, Z

    def Xp(self,times):
        lam, w, i = np.deg2rad(self.lam), np.deg2rad(self.w), np.deg2rad(self.i)
        f = self.true_anomaly(times)
        r = self.aRs*(1.-self.e**2.)/(1.+self.e*np.cos(f)) # distance
        x = -r*np.cos(f+w)*np.cos(lam) + r*np.sin(lam)*np.sin(f+w)*np.cos(i)
        return x

    def evaluate(self,times,base_error=0.):
        sigma = self.sigma
        beta = self.beta
        X = self.Xp(times)
        F = 1.-self.calc_transit(times)
        vp = X*self._Omega*np.sin(self.iS)*self.rstar*aconst.R_sun.value/1000.
        v = -1000.*vp*F*((2.*beta**2.+2.*sigma**2)/(2.*beta**2+sigma**2))**(3./2.) * (1.-(vp**2.)/(2.*beta**2+sigma**2) + (vp**4.)/(2.*(2.*beta**2+sigma**2)**2.))
        # For diagnostics
        self.vp = vp
        self.X = X
        self.F = F
        if base_error >0:
            return v + np.random.normal(loc=0.,scale=base_error,size=len(v))
        else:
            return v

def true_anomaly(time,T0,P,aRs,inc,ecc,omega):
    """
    Uses the batman function to get the true anomaly. Note that some 

    INPUT:
        time - in days
        T0 - in days
        P - in days
        aRs - in a/R*
        inc - in deg
        ecc - eccentricity
        omega - omega in deg

    OUTPUT:
        True anomaly in radians
    """
    # Some of the values here are just dummy values (limb dark etc.) to allow us to get the true anomaly
    params = batman.TransitParams()
    params.t0 = T0                           #time of inferior conjunction
    params.per = P                           #orbital period
    params.rp = 0.1                          #planet radius (in units of stellar radii)
    params.a = aRs                           #semi-major axis (in units of stellar radii)
    params.inc = inc                         #orbital inclination (in degrees)
    params.ecc = ecc                         #eccentricity
    params.w = omega                         #longitude of periastron (in degrees)
    params.u = [0.3,0.3]                     #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"           #limb darkening model
    m = batman.TransitModel(params, time)    #initializes model
    return m.get_true_anomaly()

def planet_XYZ_position(time,T0,P,aRs,inc,ecc,omega):
    """
    Get planet XYZ position

    INPUT:
        time - in days
        T0 - in days
        P - in days
        aRs - in a/R*
        inc - in deg
        ecc - eccentricity
        omega - omega in deg

    OUTPUT:
        X - planet X position
        Y - planet Y position
        Z - planet Z position

    EXAMPLE:
        Rstar = 1.
        Mstar = 1.
        inc = 90.
        ecc = 0.9
        omega = -90.
        P = 1.1
        T0 = 1.1
        Rp = 0.1
        aRs =
        print(aRs)
        x_1, y_1, z_1 = planet_XYZ_position(time,T0,P,aRs,inc,ecc,omega)
        fig, ax = plt.subplots()
        ax.plot(time,x_1)
    """
    f = true_anomaly(time,T0,P,aRs,inc,ecc,omega) # true anomaly in radiance
    omega = np.deg2rad(omega)
    inc = np.deg2rad(inc)
    r = aRs*(1.-ecc**2.)/(1.+ecc*np.cos(f)) # distance
    X = -r*np.cos(omega+f)
    Y = -r*np.sin(omega+f)*np.cos(inc)
    Z = r*np.sin(omega+f)*np.sin(inc)
    return X, Y, Z

def get_rv_curve(times_jd,P,tc,e,omega,K):
    """
    A function to calculate an RV curve as a function of time

    INPUT:
        times_jd - times in bjd
        P - period in days
        tc - transit center
        e - eccentricity
        omega - omega in degrees
        K - semi-amplitude in m/s

    OUTPUT:
        RV curve in m/s
    """
    t_peri = radvel.orbit.timetrans_to_timeperi(tc=tc,per=P,ecc=e,omega=np.deg2rad(omega))
    rvs = radvel.kepler.rv_drive(times_jd,[P,t_peri,e,np.deg2rad(omega),K])
    return rvs
