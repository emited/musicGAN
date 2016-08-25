require 'gnuplot'
require 'audio'
require 'image'

--voice = audio.samplevoice()
--print(voice:size())
--spect = audio.spectrogram(voice, 8192, 'hann', 512)
--image.display(spect)

v = audio.samplevoice()
x = torch.linspace(0, 10000, 700*1024)
t = x*math.pi*2
y = torch.sin(t)*(v:max()-v:min())+v:min()

s = audio.spectrogram(y, 8192, 'hann', 512)

image.display(s)
audio.save('test_sample.mp3', y:view(-1, 1), 11025)