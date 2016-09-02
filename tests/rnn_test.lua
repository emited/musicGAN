require 'nn'
require 'nngraph'
require 'rnn'

torch.manualSeed(122)

I = 3
H = 10

x = {}
for i = 1, 7 do	x[i] = torch.rand(4, I) end
print(x)


lstm = nn.LSTM(I, H, 1)
lstm2 = nn.LSTM(H, H, 1)
seq = nn.Sequential():add(lstm):add(lstm2)

seq_lstm = nn.Sequencer(seq)

print('normal lstm')
h0 = torch.rand(4, I)
print(seq:forward(x[1], h0))

print('without h')
print(seq:forward(x[1]))

print('sequence')
print(seq_lstm:forward(x))


print('sampling')
print(seq:forward(torch.rand(1,I)))
