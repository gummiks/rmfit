from scipy.integrate import quad
import numpy as np

def theta(r,d,rp):
    """
    Theta function from Sacket et al. 1998 (EQ 10)
    """
    return (d**2.+r**2.-rp**2.)/(2.*r*d)

def limb_dark_quadradic(x,u1,u2):
    """
    See https://astro.uchicago.edu/~kreidberg/batman/tutorial.html#limb-darkening-options
    
    INPUT:
        x  - the radius coordinate
        u1 - linear limb darkening parameter
        
    OUTPUT:
        intensity as a function of the normalized radial coordinate
        
    EXAMPLE:
        x = np.linspace(0,1,300)
        fig, ax = plt.subplots()
        ax.plot(x,limb_dark_linear(x,0.1))
        ax.plot(x,limb_dark_linear(x,0.2))
        ax.plot(x,limb_dark_linear(x,0.3))
        ax.plot(x,limb_dark_linear(x,0.9))
        
        np.trapz(limb_dark_quadradic(x,0.024315010,0.37762000),x),np.trapz(limb_dark_linear(x,0.2),x)
        np.trapz(limb_dark_quadradic(x,0.17819002,0.42043499),x),np.trapz(limb_dark_linear(x,0.3),x)
    """
    mu = np.sqrt(1.-x**2.)
    return 1.*(1.-u1*(1.-mu)-u2*((1.-mu)**2.))

def cb_limbdark(ds,rp,u1,u2,vb,epsabs=1.49e-08,epsrel=1.49e-08):
    """
    This gives the CB that is being covered by the planet (i.e., velocity of the star behind the planet)
    
    INPUT:
        d  - separation of centers of the star and the planet
        rp - radius of planet (Rp/Rs)
        u1 - Quad Limb dark parameter 1
        u2 - Quad limb dark parameter 2
        vb - Convective Blueshift velocity in m/s
        
    EXAMPLE:
        time = np.linspace(-5/24, 5/24, 200)
        inc = 90.
        ecc = 0.0
        omega = 90.
        P = 3.5
        T0 = 0.
        Rp = 0.11
        aRs = 12.
        x_1, y_1, z_1 = rmfit.planet_XYZ_position(time,T0,P,aRs,inc,ecc,omega)
        ds = np.sqrt(x_1**2.+y_1**2.)
        cb = rmfit.cb_limbdark(ds,Rp,u[0],u[1],V_CB)
    """
    def _num_integrand(r,d,rp,u1,u2,vb,baseline):
        if r > 1.:
            return 0.
        mu = np.sqrt(1.-r**2.)
        if d==0:
            return (vb*mu-baseline)*(r*np.pi*limb_dark_quadradic(r,u1,u2))
        t = theta(r,d,rp)
        if np.abs(t) > 1:
            return (vb*mu-baseline)*(r*np.pi*limb_dark_quadradic(r,u1,u2))
        else:
            return (vb*mu-baseline)*(r*np.arccos(t)*limb_dark_quadradic(r,u1,u2))

    def _denom_integrand(r,u1,u2):
        return r*limb_dark_quadradic(r,u1,u2)
    
    denominator = quad(_denom_integrand,0.,1.,args=(u1,u2),epsabs=epsabs,epsrel=epsrel)[0] # full disk
    baseline = (quad(_num_integrand,0.,1.,args=(0.,1.,u1,u2,vb,0.),epsabs=epsabs,epsrel=epsrel)[0]/np.pi) / quad(_denom_integrand,0.,1.,args=(u1,u2),epsabs=epsabs,epsrel=epsrel)[0]
    
    velocities = []
    for d in ds:
        d = np.abs(d)
        low_lim = np.max([0,d-rp])
        upp_lim = np.min([1.,d+rp])
        numerator = 0. - quad(_num_integrand,low_lim,upp_lim,args=(d,rp,u1,u2,vb,baseline),epsabs=epsabs,epsrel=epsrel)[0]/np.pi 
        velocities.append(numerator/denominator)
    return np.array(velocities)
