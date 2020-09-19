import numpy as np
import math
import pandas as pd

class PriorSet(object):
    def __init__(self, priors):
        self.priors = priors
        self.ndim  = len(priors)

        self.pmins = np.array([p.min() for p in self.priors])
        self.pmaxs = np.array([p.max() for p in self.priors])
        self.bounds= np.array([self.pmins,self.pmaxs]).T

        self.sbounds= np.array([[p.center-p.squeeze*0.5*p.width,
                                 p.center+p.squeeze*0.5*p.width] for p in self.priors])


    def generate_pv_population(self, npop):
        return np.array([[((p.random()[0]-p.center)*p.squeeze)+p.center for p in self.priors] for i in range(npop)])

    def c_log_prior(self, pv):
        return np.sum([p.log(v) for p,v in zip(self.priors, pv)])

    def c_prior(self, pv):
        return math.exp(self.c_log_prior(pv))

    @property
    def names(self):
        return [p.name for p in self.priors]
    
    # EXTENDING
    @property
    def descriptions(self):
        return [p.description for p in self.priors]
    
    @property
    def centers(self):
        return [p.center for p in self.priors]
    
    @property
    def labels(self):
        return [p.name for p in self.priors]
    
    @property
    def random(self):
        return [p.random()[0] for p in self.priors]
    
    @property
    def priortypes(self):
        return [p.priortype for p in self.priors]

    @property
    def args1(self):
        return [p.args1 for p in self.priors]

    @property
    def args2(self):
        return [p.args2 for p in self.priors]

    @property
    def ids(self):
        return [p.ID for p in self.priors]

    @property
    def fixed(self):
        return [p.fixed for p in self.priors]

    @property
    def df(self):
        return pd.DataFrame(list(zip(self.ids,self.args1,self.args2,self.labels,self.descriptions,self.priortypes,self.fixed)),
                columns=['prior','arg1','arg2','label','description','priortype','fixed'])
    
    @property
    def detrendparams(self):
        priortypes = self.priortypes

    def get_param_type_indices(self,paramtype="detrend"):
        assert (paramtype == "detrend" or paramtype=="model" or paramtype=="fixed" or paramtype=="gp")
        priortypes = self.priortypes
        df = pd.DataFrame(priortypes,columns=["priortypes"])
        return list(df[df["priortypes"] == paramtype].index)


class Prior(object):
    def __init__(self, a, b, name='', description='', unit='', squeeze=1.,priortype=""):
        self.a = float(a)
        self.b = float(b)
        self.center= 0.5*(a+b)
        self.width = b - a
        self.squeeze = squeeze

        self.name = name
        self.description = description
        self.unit = unit
        self.priortype = priortype

    def limits(self): return self.a, self.b 
    def min(self): return self.a
    def max(self): return self.b
   
 
class UniformPrior(Prior):
    def __init__(self, a, b, name='', description='', unit='', squeeze=1.,priortype="model"):
        self.args1 = a
        self.args2 = b
        self.ID = 'UP'
        super(UniformPrior, self).__init__(a,b,name,description,unit,squeeze,priortype)
        self._f = 1. / self.width
        self._lf = math.log(self._f)
        self.fixed = False

    def __call__(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b), self._f, 1e-80)
        else:
            return self._f if self.a < x < self.b else 1e-80

    def log(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b), self._lf, -1e18)
        else:
            return self._lf if self.a < x < self.b else -1e18

    def random(self, size=1):
        return np.random.uniform(self.a, self.b, size=size)


class JeffreysPrior(Prior):
    def __init__(self, a, b, name='', description='', unit='', squeeze=1.,priortype="model"):
        self.args1 = a
        self.args2 = b
        self.ID = 'JP'
        super(JeffreysPrior, self).__init__(a,b,name,description,unit,squeeze,priortype)
        self._f = math.log(b/a)
        self.fixed = False

    def __call__(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b), 1. / (x*self._f), 1e-80)
        else:
            return 1. / (x*self._f) if self.a < x < self.b else 1e-80

    def log(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b), np.log(1. / (x*self._f)), -1e18)
        else:
            return math.log(1. / (x*self._f)) if self.a < x < self.b else -1e18

    def random(self, size=1):
        return np.random.uniform(self.a, self.b, size=size)


class NormalPrior(Prior):
    def __init__(self, mean, std, name='', description='', unit='', lims=None, limsigma=5, squeeze=1,priortype="model"):
        self.args1 = mean
        self.args2 = std
        self.ID = 'NP'
        lims = lims or (mean-limsigma*std, mean+limsigma*std)
        super(NormalPrior, self).__init__(*lims, name=name, description=description, unit=unit,squeeze=squeeze,priortype=priortype)
        self.mean = float(mean)
        self.std = float(std)
        self._f1 = 1./ math.sqrt(2.*math.pi*std*std)
        self._lf1 = math.log(self._f1)
        self._f2 = 1./ (2.*std*std)
        self.fixed = False

    def __call__(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b),  self._f1 * np.exp(-(x-self.mean)**2 * self._f2), 1e-80)
        else:
            return self._f1 * exp(-(x-self.mean)**2 * self._f2) if self.a < x < self.b else 1e-80

    def log(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.where((self.a < x) & (x < self.b),  self._lf1 - (x-self.mean)**2 * self._f2, -1e18)
        else:
            return self._lf1 -(x-self.mean)**2*self._f2 if self.a < x < self.b else -1e18

    def random(self, size=1):
        return np.random.uniform(self.a, self.b, size=size) #normal(self.mean, self.std, size)
    
class FixedPrior(Prior):
    def __init__(self, value, name='', description='', unit='', lims=None, squeeze=1,priortype="model"):
        self.args1 = value
        self.args2 = value
        self.ID = 'FP'
        lims = value, value
        super(FixedPrior, self).__init__(*lims, name=name, description=description, unit=unit,squeeze=squeeze,priortype=priortype)
        self.mean = float(value)
        self.fixed = True

    def __call__(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return self.mean*np.ones(len(x))
        else:
            return self.mean

    def log(self, x, pv=None):
        return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.

    #def log(self, x, pv=None):
    #    if isinstance(x, np.ndarray):
    #        return np.where((self.a < x) & (x < self.b),  self._lf1 - (x-self.mean)**2 * self._f2, -1e18)
    #    else:
    #        return self._lf1 -(x-self.mean)**2*self._f2 if self.a < x < self.b else -1e18

    #def random(self, size=1):
    #    return np.random.uniform(self.a, self.b, size=size) #normal(self.mean, self.std, size)

class DummyPrior(Prior):
    def __init__(self, a, b, name='', description='', unit='',priortype="model"):
        super(DummyPrior, self).__init__(a, b, name=name, description=description, unit=unit,priortype="model")

    def __call__(self, x, pv=None):
        return np.ones_like(x) if isinstance(x, np.ndarray) else 1.

    def log(self, x, pv=None):
        return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.

    def random(self, size=1):
        return np.random.uniform(self.a, self.b, size=size)


UP = UniformPrior
JP = JeffreysPrior
GP = NormalPrior
NP = NormalPrior 
DP = DummyPrior
FP = FixedPrior
