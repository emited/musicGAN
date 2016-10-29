require 'gnuplot'
require 'audio'
require 'image'

voice = audio.samplevoice()
print(voice:size())
spect = audio.spectrogram(voice, 1024, 'hann', 512)
--z=audio.stft(voice, 1024, 'hann', 512)
--print(z:size())
--image.display(z[{{},{},2}])
image.display(spect)

--v = audio.samplevoice()
--x = torch.linspace(0, 10000, 700*1024)
--t = x*math.pi*2
--y = torch.sin(t)*(v:max()-v:min())+v:min()

--s = audio.spectrogram(y, 8192, 'hann', 512)

--image.display(s)
--audio.save('test_sample.mp3', y:view(-1, 1), 11025)

--t, rate = audio.load('../data/test_sine.mp3')
--image.display(audio.spectrogram(t, 1024, 'hann', 512))

