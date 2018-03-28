from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode


class Izhikevich:

    def __init__(self, cm=200., k=0.75, V_threshold=-45., V_rest=-60., V_peak=30., V_reset=-50., a=0.01, b=15.0,
                 delta_u=0.0, i_baseline=0.0):
        self.cm = cm
        self.k = k
        self.V_threshold = V_threshold
        self.V_rest = V_rest
        self.V_peak = V_peak
        self.V_reset = V_reset
        self.a = a
        self.b = b
        self.delta_u = delta_u
        self.i_baseline = i_baseline
        self.V0 = None
        self.u0 = None
        self.u = None

    def dynamical_system_equations(self, V, u, I_ext):
        dv_dt = (self.k * (V - self.V_rest) * (V - self.V_threshold) - u + I_ext + self.i_baseline) / self.cm
        du_dt = self.a * (self.b * (V - self.V_rest) - u)
        return [dv_dt, du_dt]

    def set_simulation_params(self, params):
        for k, v in params.iteritems():
            setattr(self, k, v)

    def simulate(self, dt, tstop, I_ext):
        assert self.V0 is not None and self.u0 is not None, \
            "set V0 and u0 beforehand via set_simulation_params"
        assert tstop / dt % 1 == 0
        n_timesteps = int((tstop + dt) / dt)

        V = np.zeros(n_timesteps)
        t = np.zeros(n_timesteps)
        u = np.zeros(n_timesteps)
        V[0] = self.V0
        u[0] = self.u0

        def f(ts, x):
            V_ = x[0]
            u_ = x[1]
            derivatives = self.dynamical_system_equations(V_, u_, I_ext(ts))
            return derivatives

        ode_solver = ode(f)
        ode_solver.set_integrator('vode', rtol=1e-8, atol=1e-8, method='bdf')
        ode_solver.set_initial_value([V[0], u[0]], 0)

        i = 1
        while i < n_timesteps:
            assert ode_solver.successful()
            if V[i-1] >= self.V_peak:
                V[i-1] = self.V_peak
                V[i] = self.V_reset
                u[i] = u[i-1] + self.delta_u
                t[i] = t[i-1] + dt
            else:
                ode_solver.set_initial_value([V[i-1], u[i-1]], t[i-1])
                # force the integrator to start a new integration at each time step (account for discontinuities in I_ext)
                sol = ode_solver.integrate(t[i-1] + dt)

                V[i] = sol[0]
                t[i] = ode_solver.t
                u[i] = sol[1]
            i += 1
        self.u = u
        return V, t

    def simulate_foward_euler(self, dt, tstop, i_inj):
        assert self.V0 is not None and self.u0 is not None, \
            "set V0 and u0 beforehand via set_simulation_params"
        t = np.arange(0, tstop + dt, dt)
        i_inj_array = np.array([i_inj(ts) for ts in t])
        v = np.zeros(len(t))
        u = np.zeros(len(t))
        v[0] = self.V0
        u[0] = self.u0

        for i in range(0, len(t) - 1):
            v[i + 1] = v[i] + dt * (self.k * (v[i] - self.V_rest) * (v[i] - self.V_threshold)
                                    - u[i] + i_inj_array[i] + self.i_baseline) / self.cm
            u[i + 1] = u[i] + dt * (self.a * (self.b * (v[i] - self.V_rest) - u[i]))
            if v[i] >= self.V_peak:
                v[i] = self.V_peak
                v[i + 1] = self.V_reset
                u[i + 1] += self.delta_u

        if v[-1] >= self.V_peak:
            v[-1] = self.V_peak
        self.u = u
        return v, t

    def phase_plot(self, vmin=-100., vmax=100., umin=-100., umax=100., i_inj=0., v_trajectory=None, u_trajectory=None):
        V, U = np.meshgrid(np.arange(vmin, vmax+5, 5), np.arange(umin, umax+5, 5))
        dvdt = (self.k * (V - self.V_rest) * (V - self.V_threshold) - U + i_inj) / self.cm
        dudt = (self.a * (self.b * (V - self.V_rest) - U))

        def vcline(v):
            u = self.k * (v - self.V_rest) * (v - self.V_threshold) + i_inj
            return u

        def ucline(v):
            u = self.b * (v - self.V_rest)
            return u

        v_range = np.arange(vmin, vmax, 0.1)

        plt.figure()
        plt.title('Phase Plot')
        plt.quiver(V, U, dvdt, dudt, color='k', angles='xy', scale_units='xy', scale=2.5)
        if v_trajectory is not None and u_trajectory is not None:
            plt.plot(v_trajectory, u_trajectory, 'g')
        plt.plot(v_range, vcline(v_range), '-r')
        plt.plot(v_range, ucline(v_range), '-b')
        plt.xlim([vmin, vmax])
        plt.ylim([umin, umax])
        plt.xlabel('V')
        plt.ylabel('u')
        plt.show()


if __name__ == '__main__':

    # cell = Izhikevich(1, 0.04, -5./0.04, 0, 30, -60, 1, 0.2, -21, 140)  # DAP Model of the Izhikevich Book
    # cell.set_simulation_params({'V0': -70, 'u0': -70 * cell.b, })
    # tstop = 100  # ms
    # dt = 0.1  # ms
    # I_ext = lambda x: 20 if 10 <= x <= 12 else 0  # pA
    # v, t = cell.simulate(dt, tstop, I_ext)
    # v, t = cell.simulate_foward_euler(dt, tstop, I_ext)
    # plt.figure()
    # plt.plot(t, v)
    # cell.phase_plot(-80, cell.V_peak + 5, -30, 10, cell.i_baseline, v, cell.u)


    cell = Izhikevich()
    cell.set_simulation_params({'V0': -60, 'u0': 0, })

    tstop = 1400  # ms
    dt = 0.01  # ms
    I_ext = lambda x: -500 if 100 <= x <= 1200 else 0  # pA

    v, t = cell.simulate(dt, tstop, I_ext)
    #v, t = cell.simulate_foward_euler(tstop, dt, I_ext, V0, u0)

    plt.figure()
    plt.plot(t, v)
    cell.phase_plot(-80, cell.V_peak + 5, -100, 100, cell.i_baseline, v, cell.u)
