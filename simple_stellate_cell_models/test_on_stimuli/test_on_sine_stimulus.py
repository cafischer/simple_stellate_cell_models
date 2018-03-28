import matplotlib.pyplot as pl
from simple_stellate_cell_models.test_on_stimuli import get_sine_stimulus
from simple_stellate_cell_models.leaky_integrate_and_fire import LIF
from simple_stellate_cell_models.izhikevich import Izhikevich

if __name__ == '__main__':
    lif_cell = LIF(cm=100, E_pas=-70, V_threshold=-45.0, V_reset=-52.0, g_pas=0.1, delta_g_ADP=0.75, delta_g_AHP=0.5, tau_ADP=1.5, tau_AHP=35)
    # rampIV fit: LIF(cm=100, E_pas=-70, V_threshold=-45, V_reset=-52.0, g_pas=0.1, delta_g_ADP=3.0, delta_g_AHP=2.0, tau_ADP=1.5, tau_AHP=35)
    # sine fit: lif_cell = LIF(cm=100, E_pas=-70, V_threshold=-45.0, V_reset=-52.0, g_pas=0.1, delta_g_ADP=0.75, delta_g_AHP=0.5, tau_ADP=1.5, tau_AHP=35)
    # stim: I_ext = get_sine_stimulus(30.0, 30.0, dur1, 5, onset)
    # model of Higgs: lif_cell = LIF()
    lif_cell.set_simulation_params({'V0': -70, 'g_ADP0': 0, 'g_AHP0': 0})

    izhikevich_cell = Izhikevich()
    izhikevich_cell.set_simulation_params({'V0': -60, 'u0': 0,})


    cells = [lif_cell, izhikevich_cell]

    # stim
    onset = 100
    dur1 = 2000
    freq2=5
    tstop = 2*onset + dur1
    dt = 0.01
    amp1 = 30.0
    amp2 = 30.0
    I_ext = get_sine_stimulus(amp1, amp2, dur1, freq2, onset)

    # simulation
    Vs = []
    ts = []
    for cell in cells:
        V, t = cell.simulate(dt, tstop, I_ext)
        Vs.append(V)
        ts.append(t)


    fig, axes = pl.subplots(len(cells), 1, sharey='all', sharex='all')
    for i, (t, V) in enumerate(zip(ts, Vs)):
        axes[i].plot(t, V)
    pl.show()

    # pl.figure()
    # pl.plot(t, [I_ext(ts) for ts in t])
    # pl.show()

