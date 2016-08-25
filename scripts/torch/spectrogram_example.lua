require 'audio'
require 'image'

voice = audio.samplevoice()
print(voice:size())
--spect = audio.spectrogram(voice, 8192, 'hann', 512)
--image.display(spect)
