import numpy as np
from scipy.integrate import ode
from lmfit import minimize, Parameters

class SIRXConfirmedModel:

    def __init__(self):
        pass

    # set equation of motion for SIRX dynamics
    def dxdt(self,t,y,eta,rho,kappa,kappa0):

        S = y[0]
        I = y[1]
        X = y[2]
        H = y[3]

        dy = np.zeros(4)
        dy[0] = -eta*S*I - kappa0*S
        dy[1] = +eta*S*I - rho*I - kappa*I - kappa0*I
        dy[2] = +kappa*I + kappa0*I
        dy[3] = +kappa0*S


        return dy

    def SIRX(self,t, y0, eta, rho, kappa,kappa0, N, I0_factor, S0ref):

        X0 = y0 / N
        I0 = X0 * I0_factor
        S0 = S0ref-X0-I0
        y0 = np.array([S0, I0, X0, 0.0])
        t0 = t[0]

        t = t[1:]

        r = ode(self.dxdt)

        # Runge-Kutta with step size control
        r.set_integrator('dopri5')

        # set initial values
        r.set_initial_value(y0,t0)

        # set transmission rate and recovery rate
        r.set_f_params(eta,rho,kappa,kappa0)

        result = np.zeros((4,len(t)+1))
        result[:,0] = y0

        # loop through all demanded time points
        for it, t_ in enumerate(t):

            # get result of ODE integration
            y = r.integrate(t_)

            # write result to result vector
            result[:,it+1] = y

        return result

    def residual(self,params, x, data):

        eta = params['eta']
        rho = params['rho']
        kappa = params['kappa']
        kappa0 = params['kappa0']
        I0_factor = params['I0_factor']
        S0ref = params['S0']
        #N = 10**params['log10N']
        N = params['N']

        result = self.SIRX(x, data[0], eta, rho, kappa, kappa0, N, I0_factor, S0ref)
        X = result[2,:]

        residual = X*N - data

        return residual

    def fit(self,t, data,max_nfev=100000,params=None,N=None,Nmax=None,method='leastsq',**kwargs):

        if params is None:
            params = Parameters()
            R0 = kwargs.get('R0', 6.2)
            rho = kwargs.get('rho', 1/8)
            S0 = kwargs.get("S0", 1.0)
            eta = R0*rho
            params.add('eta',value=eta, vary=False)
            params.add('rho',value=rho, vary=False)
            params.add('S0', value=S0, vary=False)
            params.add('kappa',value=rho,min=0,max=1)
            params.add('kappa0',value=rho/1000,min=0,max=1.)
            #params.add('I0_factor',value=10,min=0.001)
            params.add('I0_factor',value=2., min=0)
            varyN = N is None
            if varyN:
                N = 1e7
            if Nmax is None:
                Nmax=115000000
            params.add('N',value=N,min=1000,max=Nmax,vary=varyN)

        out = minimize(self.residual, params, args=(t, data, ),
                       max_nfev=max_nfev,
                       method=method,
                )
        return out
