from __future__ import division
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as pl


class LIF:

    def __init__(self, cm=500., g_pas=20., delta_g_ADP=20., delta_g_AHP=5., tau_ADP=1., tau_AHP=50.,
                 E_pas=-80., E_ADP=70., E_AHP=-100., V_threshold=-55., V_reset=-60., V_peak=30., dt_refractory=2.):
        self.cm = cm  # pF
        self.g_pas = g_pas  # nS
        self.delta_g_ADP = delta_g_ADP  # nS/spikes
        self.delta_g_AHP = delta_g_AHP  # nS/spikes
        self.tau_ADP = tau_ADP  # ms
        self.tau_AHP = tau_AHP  # ms
        self.E_pas = E_pas  # mV
        self.E_ADP = E_ADP  # mV
        self.E_AHP = E_AHP  # mV
        self.V_threshold = V_threshold  # mV
        self.V_reset = V_reset  # mV
        self.V_peak = V_peak
        self.dt_refractory = dt_refractory  # 2 ms
        self.V0 = None
        self.g_ADP0 = None
        self.g_AHP0 = None
        self.g_ADP = None
        self.g_AHP = None

    def dynamical_system_equations(self, V, g_ADP, g_AHP, I_ext):
        dg_ADP_dt = -g_ADP / self.tau_ADP
        dg_AHP_dt = -g_AHP / self.tau_AHP

        I_m = self.g_pas * (V - self.E_pas) + g_ADP * (V - self.E_ADP) + g_AHP * (V - self.E_AHP)
        dV_dt = (I_ext - I_m) / self.cm
        return [dV_dt, dg_ADP_dt, dg_AHP_dt]

    def set_simulation_params(self, params):
        for k, v in params.iteritems():
            setattr(self, k, v)

    def simulate(self, dt, tstop, I_ext):
        assert self.V0 is not None and self.g_ADP0 is not None and self.g_AHP0 is not None, \
            "set V0, g_ADP0 and g_AHP0 beforehand via set_simulation_params"
        assert tstop/dt % 1 == 0
        assert self.dt_refractory/dt % 1 == 0
    
        n_timesteps = int((tstop+dt) / dt)
        n_refractory = int(self.dt_refractory / dt)
    
        V = np.zeros(n_timesteps)
        t = np.zeros(n_timesteps)
        g_ADP = np.zeros(n_timesteps)
        g_AHP = np.zeros(n_timesteps)
        V[0] = self.V0
        g_ADP[0] = self.g_ADP0
        g_AHP[0] = self.g_AHP0
    
        def f(ts, x):
            V_ = x[0]
            g_ADP_ = x[1]
            g_AHP_ = x[2]
            derivatives = self.dynamical_system_equations(V_, g_ADP_, g_AHP_, I_ext(ts)*1000)  # I_ext from nA to pA
            return derivatives
    
        ode_solver = ode(f)
        ode_solver.set_integrator('vode', rtol=1e-8, atol=1e-8, method='bdf')
        ode_solver.set_initial_value([V[0], g_ADP[0], g_AHP[0]], 0)
    
        i = 1
        counter_refractory = 0
        while i < n_timesteps:
            assert ode_solver.successful()
            if counter_refractory != 0:
                ode_solver.set_initial_value([self.V_reset, g_ADP[i-1], g_AHP[i-1]], t[i-1])
    
                if counter_refractory == 1:
                    ode_solver.set_initial_value([self.V_reset, g_ADP[i-1] + self.delta_g_ADP,
                                                  g_AHP[i-1] + self.delta_g_AHP], t[i-1])
                counter_refractory -= 1
            else:
                ode_solver.set_initial_value([V[i-1], g_ADP[i-1], g_AHP[i-1]], t[i-1])
                # force the integrator to start a new integration at each time step (account for discontinuities in I_ext)
            sol = ode_solver.integrate(ode_solver.t + dt)
            if counter_refractory != 0:
                V[i] = self.V_reset
            else:
                V[i] = sol[0]
            t[i] = ode_solver.t
            g_ADP[i] = sol[1]
            g_AHP[i] = sol[2]
    
            if V[i] >= self.V_threshold and counter_refractory == 0:
                n_refractory_tmp = n_refractory if i + n_refractory < n_timesteps else n_timesteps - i - 1
                counter_refractory = n_refractory_tmp
                V[i] = self.V_peak  # to have a visible spike
    
            i += 1
        self.g_ADP = g_ADP
        self.g_AHP = g_AHP
        return V, t


if __name__ == '__main__':
    lif_cell = LIF(dt_refractory=1, delta_g_ADP=20, delta_g_AHP=5)
    lif_cell.set_simulation_params({'V0': -80, 'g_ADP0': 0, 'g_AHP0': 0})

    dt = 0.01
    tstop = 100
    I_ext = lambda x: 1.7 if 5 <= x <= 15 else 0  # pA
    #I_ext = lambda x: 0.8 if 10 <= x <= 100 else 0  # pA
    V, t = lif_cell.simulate(dt, tstop, I_ext)


    pl.figure()
    pl.plot(t, V)
    #pl.show()

    pl.figure()
    pl.plot(t, lif_cell.g_ADP, c='r')
    pl.plot(t, lif_cell.g_AHP, c='b')
    pl.show()
