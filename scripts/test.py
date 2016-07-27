# -*- coding: utf-8 -*-
import scipy, numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

sigrate, sig = wavfile.read('prelude_mono.wav')
secs = float(len(sig))/sigrate
t=np.linspace(0, secs, len(sig))
plt.plot(t, sig)

print np.fft.fft(sig[:10000])