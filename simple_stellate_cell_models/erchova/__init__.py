from __future__ import division
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as pl
from simple_stellate_cell_models.test_on_stimuli import get_zap_stimulus
from cell_characteristics import to_idx


def fun_for_fitting(f, c, R_l, R, L):
    impedance = 1. / c * np.sqrt(((2 * np.pi * f) ** 2 * L ** 2 + R_l ** 2)
                                 / ((L / (R * c) + R_l) ** 2 * (2 * np.pi * f) ** 2
                                    + (1 / c * (1 + R_l / R) - (2 * np.pi * f) ** 2 * L) ** 2))
    return impedance


class Erchova:
    def __init__(self, c=3.635e-4, R_l=46.665, R=51.986, L=2470608.308,
                 V_threshold=-45., V_peak=30., V_reset=-52., V_rest=-70., dt_refractory=2.):
        self.c = c * 1000  # uF * 1000 = mF
        self.R_l = R_l  # MOhm
        self.R = R  # MOHm
        self.L = L / 1000  # H (Ohm*sec) / 1000 = (Ohm*ms)
        self.V0 = None
        self.I_l0 = None
        self.I_l = None
        self.V_threshold = V_threshold
        self.V_peak = V_peak
        self.V_reset = V_reset
        self.V_rest = V_rest
        self.dt_refractory = dt_refractory
        self.V0 = None
        self.I_l0 = None

    def dynamical_system_equations(self, V, I_l, I_ext):
        dI_l_dt = (-self.R_l * I_l + V) / self.L
        dV_dt = (-1./self.R * V - I_l + I_ext) / self.c
        return [dV_dt, dI_l_dt]

    def set_simulation_params(self, params):
        for k, v in params.iteritems():
            setattr(self, k, v)

    def simulate(self, dt, tstop, I_ext):
        assert self.V0 is not None and self.I_l0 is not None, \
            "set V0 and I_l0 beforehand via set_simulation_params"
        assert tstop / dt % 1 == 0
        n_timesteps = int((tstop + dt) / dt)

        V = np.zeros(n_timesteps)
        t = np.zeros(n_timesteps)
        I_l = np.zeros(n_timesteps)
        V[0] = self.V0
        I_l[0] = self.I_l0
        n_refractory = to_idx(self.dt_refractory, dt)

        def f(ts, x):
            V_ = x[0]
            I_l_ = x[1]
            derivatives = self.dynamical_system_equations(V_, I_l_, I_ext(ts))
            return derivatives

        ode_solver = ode(f)
        ode_solver.set_integrator('vode', rtol=1e-8, atol=1e-8, method='bdf')
        ode_solver.set_initial_value([V[0], I_l[0]], 0)

        i = 1
        while i < n_timesteps:
            # TODO assert ode_solver.successful()
            # if not ode_solver.successful():
            #     break
            if V[i-1] >= (self.V_threshold - self.V_rest):  # model is set up to be around 0 not V_rest
                V[i-1] = (self.V_peak - self.V_rest)
                V[i:i+n_refractory] = (self.V_reset - self.V_rest)
                I_l[i:i+n_refractory] = I_l[i-1]
                t[i:i+n_refractory] = np.arange(i, i+n_refractory, 1) * dt
                i += n_refractory - 1
            else:
                ode_solver.set_initial_value([V[i-1], I_l[i-1]], t[i-1])
                # force the integrator to start a new integration at each time step (account for discontinuities in I_ext)
                sol = ode_solver.integrate(t[i-1] + dt)

                V[i] = sol[0]
                t[i] = ode_solver.t
                I_l[i] = sol[1]
            i += 1
        self.I_l = I_l
        return V + self.V_rest, t


if __name__ == '__main__':
    # # fit parameters to impedance curve
    # impedance_exp = np.load('./impedance.npy')
    # frequencies = np.load('./frequencies.npy')
    # impedance_to_fit = impedance_exp * 1e6  # convert impedance to Ohm
    # p_opt, _ = curve_fit(fun_for_fitting, frequencies, impedance_to_fit, p0=[4e-10, 20e6, 20e6, 1e6],
    #                      bounds=[0, np.inf])
    # c, R_l, R, L = p_opt
    # c = c * 1e6  # uF
    # R_l = R_l * 1e-6  # MOhm
    # R = R * 1e-6  # MOhm
    # L = L  # Ohm * sec
    # print c, R_l, R, L
    #
    # pl.figure()
    # pl.plot(frequencies, impedance_to_fit, label='exp.')
    # pl.plot(frequencies, fun_for_fitting(frequencies, c, R_l, R, L), label='fit')
    # pl.ylabel('Impedance (MOhm)')
    # pl.xlabel('Frequency (Hz)')
    # pl.legend()
    # pl.show()

    cell = Erchova()  # Erchova(c, R_l, R, L)
    cell.set_simulation_params({'V0': 0, 'I_l0': 0})

    dt = 0.1

    tstop = 100
    I_ext = lambda x: 500 if 10 <= x <= 12 else 0  # pA
    V, t = cell.simulate(dt, tstop, I_ext)

    pl.figure()
    pl.title('Ramp')
    pl.plot(t, V)
    pl.show()

    tstop = 700
    I_ext = lambda x: 0.300 if 100 <= x <= 600 else 0  # pA
    V, t = cell.simulate(dt, tstop, I_ext)

    pl.figure()
    pl.title('Step')
    pl.plot(t, V)
    #pl.show()

    onset = 100
    dur_zap = 30000
    tstop = dur_zap + 2 * onset
    freq0 = 0
    freq1 = 20
    I_ext = get_zap_stimulus(amp=0.1, freq0=freq0, freq1=freq1, onset=onset, dur_zap=dur_zap)  #30000

    V, t = cell.simulate(dt, tstop, I_ext)

    pl.figure()
    pl.title('ZAP')
    pl.plot(t, V)
    pl.show()

    # # cut off onset and offset and downsample
    # # ds = 10  # number of steps skipped (in t, i, v) for the impedance computation
    # # t_ds = t[to_idx(onset, dt, 3):to_idx(onset + dur_zap, dt, 3):ds]
    # # i_inj_ds = np.array([I_ext(ts) for ts in t])[to_idx(onset, dt, 3):to_idx(onset + dur_zap, dt, 3):ds]
    # # v_ds = V[to_idx(onset, dt, 3):to_idx(onset + dur_zap, dt, 3):ds]
    #
    # v_ds = V[to_idx(onset, dt, 3):to_idx(onset + dur_zap, dt, 3)]
    # t_ds = t[to_idx(onset, dt, 3):to_idx(onset + dur_zap, dt, 3)] - onset
    # i_inj_ds = np.array([I_ext(ts) for ts in t])[to_idx(onset, dt, 3):to_idx(onset + dur_zap, dt, 3)]
    #
    # # compute impedance
    # imp, frequencies2, imp_smooth = impedance(v_ds, i_inj_ds, (t_ds[1] - t_ds[0]) / 1000,
    #                                          [freq0, freq1])  # dt in (sec) for fft
    #
    # pl.figure()
    # pl.plot(frequencies, impedance_exp, label='exp.')
    # pl.plot(frequencies, fun_for_fitting(frequencies, *p_opt) * 1e-6, label='fit')
    # pl.plot(frequencies2, imp_smooth, label='model')
    # pl.legend()
    # pl.ylabel('Impedance (MOhm)')
    # pl.xlabel('Frequency (Hz)')
    # pl.show()