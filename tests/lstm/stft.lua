require 'audio'
require 'image'
local signal = require 'signal'

function audio.istft(X, win, hop)
    local x = torch.zeros((X:size(1)-1)*hop + win)
    framesamp = X:size(2)
    hopsamp = hop
    for n=1,X:size(1) do
        i = 1 + (n-1)*hopsamp
        --print(i, i + framesamp - 1)
        x[{{i, i + framesamp - 1}}] = x[{{i, i + framesamp - 1}}]
        	+ signal.complex.real(signal.ifft(X[n]))
    end
    return x
end

torch.setdefaulttensortype('torch.FloatTensor')

inp = audio.samplevoice():float():squeeze()
print(#(inp))

--signal stft
--stft = signal.stft(inp, 1024, 512, 'hamming')
--stft = signal.stft(inp, 1024, 512)
--stft = signal.stft(inp, 1024, 512, 'bartlett')
a=os.clock()
stft = signal.stft(inp, 1024, 512, 'hann')
print(stft:size())
print((inp:size(1)-512)/1024*2)
print('Time taken for stft from signal package: ' .. os.clock()-a)

--audio stft
--a=os.clock()
--stft2 = audio.stft(inp, 1024, 'hann', 512)
--print('Time taken for stft from audio package: ' .. os.clock()-a)
--print(#stft)

--istft
a=os.clock()
istft = audio.istft(stft, 1024, 512)
print('Time taken for istft: ' .. os.clock()-a)
audio.save('voice_istft.mp3', istft:view(-1,1), 22050)
print(istft:size())


-- display magnitude
--image.display(stft[{{1,100},{1,100},1}])
--image.display(stft2[{{1,100},{stft2:size(2)-100,stft2:size(2)},1}])
--
--spect =  signal.spectrogram(inp, 1024, 512)
--image.display(spect)
