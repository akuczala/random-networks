import numpy as np
import scipy.optimize as opt
from scipy import linalg as lin
from scipy import integrate as nint
from scipy.interpolate import interp1d
import sys
this = sys.modules[__name__]

def spectral_abscissa_G(G):
    return np.flip(np.sort(np.real(lin.eigvals(G))),axis=0)[0]

def phi(x): #nonlinearity
    return np.tanh(x)
def dphi(x): #derivative of nonlinearity
    return 1 - np.tanh(x)**2

#compute expectation value of f(x) given q
def calc_q_vev(f,q,epsrel=1e-8,**kwargs):
    #if q is zero, we don't need to do the integral
    if np.isclose(q,0):
        return f(0)
    #absolute value prevents invalid argument
    return nint.quad(lambda x: np.exp(-x*x/2)*f(np.sqrt(np.abs(q))*x)/np.sqrt(2*np.pi),
                     -np.inf,np.inf,epsrel=epsrel)[0]

#compute expectation values of f(x_k) given q_k
def calc_qs_vev(f,qvec,**kwargs):
    return np.array([calc_q_vev(f,q,**kwargs) for q in qvec])

#get possibly precomputed expectation values of f(x_k) given q_k
def get_fn_vevs(fn_name,qvec,**kwargs):
    return np.array([get_fn_q_vev(fn_name,q,**kwargs) for q in qvec])

#RHS of self consistent equations for q_k
def self_consistent_qs(qvec,G,sigma_internal = 0,**kwargs):
    return sigma_internal**2 + np.dot(G,get_fn_vevs('phisq',qvec,**kwargs))

#compute q_k by finding the fixed point of the self_consistent equations
def calc_qs(G,sigma_internal=0,**kwargs):
    R = spectral_abscissa_G(G)
    if R < 1 and sigma_internal == 0:
        return np.zeros([len(G)])
    q0 = R*np.ones(len(G))
    #abs prevents negative q solutions
    sol = opt.root(lambda qvec: self_consistent_qs(qvec,G,sigma_internal=sigma_internal,**kwargs) - qvec,
                   q0,method='lm')
    if not sol.success:
       print("couldn't solve for qs",G,sol.x,self_consistent_qs(sol.x,G,**kwargs))
       raise
    return sol.x

#calculate effective gain matrix
def calc_M(G,qvec,**kwargs):
    return G*(get_fn_vevs('dphi',qvec,**kwargs)**2)

#fisher memory curve for given t range
def info_t(t_range,G,fracs,w,sig_obs,**kwargs):
    qvec = calc_qs(G,**kwargs)
    M = calc_M(G,qvec,**kwargs)
    
    noise_fac = (fracs/(sig_obs**2 + qvec))[:,np.newaxis]
    return np.array([ np.sum(noise_fac*np.linalg.matrix_power(M,t)*w**2) for t in t_range])

#calculate fisher information over all t>=0
def calc_info(G,fracs,w,sig_obs,qvec=None,M=None,**kwargs):
    if M is None:
        if qvec is None:
            qvec = calc_qs(G,**kwargs)
        M = calc_M(G,qvec,**kwargs)
    noise_fac = (fracs/(sig_obs**2 + qvec))[:,np.newaxis]
    return np.sum(noise_fac*np.dot(M,lin.inv(np.eye(len(M))-M))*w**2)

#calculate fisher information over all t>0, with optimal choice of w
def calc_max_info(G,fracs,sig_obs,qvec=None,M=None,**kwargs):
    if M is None:
        if qvec is None:
            qvec = calc_qs(G,**kwargs)
        M = calc_M(G,qvec,**kwargs)
    
    noise_fac = (fracs/(sig_obs**2 + qvec))[:,np.newaxis]
    return lin.norm(noise_fac*np.dot(M,lin.inv(np.eye(len(M))-M)),1)

#single population fisher information
def calc_single_info(g,**kwargs):
    return calc_max_info(np.array([[g*g]]),np.array([1]),**kwargs)

#calculate lypunov exponent    
def calc_lyapunov(G,**kwargs):
    qs = calc_qs(G,**kwargs)
    M_somp = G*get_fn_vevs('dphisq',qs,**kwargs)
    Meigs_min = np.real(np.min(lin.eigvals(np.eye(len(G))-M_somp)))
    return -1 + np.sqrt(1-Meigs_min)

#compute interpolating functions for expectation values of f in q_range
def calc_interp_q_vev(f,q_range):
    return interp1d(q_range,[calc_q_vev(f,q) for q in q_range],kind='quadratic')

#get value of interpolating function,
#or calculate if either q is not in the interpolation range,
#or interpolation is turned off
def get_fn_q_vev(fn_name,q,interp=False,**kwargs):
    if interp and q >= this.q_min and q <= this.q_max:
        return this.interp_dict[fn_name](q)
    else:
        return calc_q_vev(this.fn_dict[fn_name],q,**kwargs)

#compute interpolating functions for frequently used expectation values
def calc_interp_fns(q_min = 0, q_max = 4, interpolation_pts = 2000):
    print("Computing interpolating functions...")
    this.q_min = q_min; this.q_max = q_max
    this.interpolation_pts = interpolation_pts
    this.q_range = np.linspace(q_min,q_max,interpolation_pts)

    this.fn_dict = {
        "phisq" : lambda x: phi(x)**2,
        "dphi" : dphi,
        "dphisq" : lambda x: dphi(x)**2}
    #compute interpolation fn for each fn in fn_dict
    this.interp_dict = {fn_name : calc_interp_q_vev(fn,this.q_range)
        for fn_name, fn in this.fn_dict.items()}

    print("done.")

#precompute interpolation functions
calc_interp_fns()

