import wavelets
from wavelets import WaveletAnalysis
import numpy as np
from scipy.io import wavfile

# given a signal x(t)
fs, data = wavfile.read('voice.wav')
#print('ok wavfile')
x = data
#fs = 300
#x = np.random.randn(1000)
# and a sample spacing
dt = 0.1

wa = WaveletAnalysis(data=x, wavelet=wavelets.Ricker(),dt=dt)

# wavelet power spectrum
power = wa.wavelet_power

# scales 
scales = wa.scales

# associated time vector
t = wa.time

# reconstruction of the original data
rx = wa.reconstruction()
print(np.real(rx))
wavfile.write('reconstruction.wav', fs, np.real(rx))
#import matplotlib.pyplot as plt

#fig, ax = plt.subplots()
#T, S = np.meshgrid(t, scales)
#ax.contourf(T, S, power, 100)
#ax.set_yscale('log')
#fig.savefig('test_wavelet_power_spectrum.png')
