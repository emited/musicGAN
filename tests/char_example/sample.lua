
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'

opt = {
	vocabfile = 'vocab.t7',
	model = 'char_lstm_epoch_800.t7',
	seed = 126,
	sample = true,
	primetext = 'hello my name is ',
	length = 300,
}

torch.manualSeed(opt.seed)
local vocab = torch.load(opt.vocabfile)
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- load model and recreate a few important numbers
protos = torch.load(opt.model)
opt.rnn_size = protos.embed.weight:size(2)

--protos.embed = Embedding(vocab_size, opt.rnn_size)
---- lstm timestep's input: {x, prev_c, prev_h}, output: {next_c, next_h}
--protos.lstm = LSTM.lstm(opt)
--protos.softmax = nn.Sequential():add(nn.Linear(opt.rnn_size, vocab_size)):add(nn.LogSoftMax())
--protos.criterion = nn.ClassNLLCriterion()

-- LSTM initial state, note that we're using minibatches OF SIZE ONE here
local prev_c = torch.zeros(1, opt.rnn_size)
local prev_h = prev_c:clone()

local seed_text = opt.primetext
local prev_char

-- do some seeded timesteps
for c in seed_text:gmatch'.' do
    prev_char = torch.Tensor{vocab[c]}

    local embedding = protos.embed:forward(prev_char)
    local next_c, next_h = unpack(protos.lstm:forward{embedding, prev_c, prev_h})

    prev_c:copy(next_c) -- TODO: this shouldn't be needed... check if we can just use an assignment?
    prev_h:copy(next_h)
end

-- now start sampling/argmaxing
for i=1, opt.length do
    -- embedding and LSTM 
    local embedding = protos.embed:forward(prev_char)
    local next_c, next_h = unpack(protos.lstm:forward{embedding, prev_c, prev_h})
    prev_c:copy(next_c)
    prev_h:copy(next_h)
    
    -- softmax from previous timestep
    local log_probs = protos.softmax:forward(next_h)

    if not opt.sample then
        -- use argmax
        local _, prev_char_ = log_probs:max(2)
        prev_char = prev_char_:resize(1)
    else
        -- use sampling
        local probs = torch.exp(log_probs):squeeze()
        prev_char = torch.multinomial(probs, 1):resize(1)
    end

    --print('OUTPUT:', ivocab[prev_char[1]])
    io.write(ivocab[prev_char[1]])
end
io.write('\n') io.flush()