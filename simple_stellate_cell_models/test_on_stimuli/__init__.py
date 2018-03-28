from __future__ import division
import numpy as np


def get_sine_stimulus(amp1, amp2, dur1, freq2, onset):
    """
    im Laborbuch: amp1, amp2, freq2, stim_dur
    :param amp1: amplitude of underlying sine in nA
    :param amp2: amplitude of modulating sine in nA
    :param freq2: in Hz
    :param dur1: duration of big sine in ms
    :return: sine stimulus
    """
    freq2 = freq2 / 1000  # per ms
    freq1 = 1 / dur1 / 2  # per ms (only half a sine)

    def sine_stim(ts):
        if ts < onset or ts > onset + dur1:
            return 0
        else:
            sine1 = np.sin(2 * np.pi * (ts-onset) * freq1)
            sine2 = np.sin(2 * np.pi * (ts-onset) * freq2)
            sine_sum = amp1 * sine1 + amp2 * sine2
            return sine_sum
    return sine_stim


def get_zap_stimulus(amp=0.1, freq0=0, freq1=20, onset=2000, dur_zap=30000):
    def zap_stim(ts):
        if ts < onset or ts > onset + dur_zap:
            return 0
        else:
            return amp * np.sin(2 * np.pi * ((freq1 - freq0) / 1000 * (ts-onset) / (2 * dur_zap) + freq0/1000) * (ts-onset))
    return zap_stim