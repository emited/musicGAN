import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.io import wavfile


fs, data = wavfile.read('voice.wav')
print(fs)
print(pywt.wavelist())

(cA, cD) = pywt.dwt(data,'bior2.4',mode='sp1')
A = pywt.idwt(cA, cD, 'bior2.4', 'sp1')
print('init data length = '+str(len(data)))
print('len='+str(len(cA))+' len2='+str(len(cD))+' sum = '+str(len(cA)+len(cD)))

wavfile.write('recons.wav', fs, A)
print(A)
