
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
local CharLMMinibatchLoader = require 'CharLMMinibatchLoader'
local LSTM = require 'LSTM'
require 'Embedding'
local model_utils = require 'model_utils'

opt = {
	--data
	vocabfile = 'vocab.t7',
	datafile = 'train.t7',
	batch_size = 16,

	--model
	seq_length = 16,
	input_size = 256,
	rnn_size = 256,
	max_epochs = 10,
	seed = 123,

	--misc
	savefile = 'char_lstm',
	save_every = 100,
	print_every = 1,
}

torch.manualSeed(opt.seed)

local loader = CharLMMinibatchLoader.create(
	opt.datafile, opt.vocabfile, opt.batch_size, opt.seq_length)

local vocab_size = loader.vocab_size
print('vocab size = '..vocab_size)




local protos = {}
protos.embed = Embedding(vocab_size, opt.rnn_size)
protos.lstm = LSTM.lstm(opt)
protos.softmax = nn.Sequential():add(nn.Linear(opt.rnn_size, vocab_size)):add(nn.LogSoftMax())
protos.criterion = nn.ClassNLLCriterion()

local x,y = loader:next_batch()
print('x:')
print(x:size())
print(y:size())
print(x) ; print(y)


local params, grad_params = model_utils.combine_all_parameters(protos.embed, protos.lstm, protos.softmax)
params:uniform(-0.08, 0.08)

local clones = {}
for name, proto in pairs(protos) do
	print('cloning '..name)
	clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

local initstate_c = torch.zeros(opt.batch_size, opt.rnn_size)
local initstate_h = initstate_c:clone()

local dfinalstate_c = initstate_c:clone()
local dfinalstate_h = initstate_c:clone()


function feval(params_)
	if params_ ~= params then
		params:copy(params_)
	end
	grad_params:zero()

	--forward pass
	local x,y = loader:next_batch()
	local embeddings = {}
	local lstm_c = {[0]=initstate_c}
	local lstm_h = {[0]=initstate_h}
	local predictions = {}
	local loss = 0

	for t = 1, opt.seq_length do
		embeddings[t] = clones.embed[t]:forward(x[{{}, t}])
		--print('okokokokokokok')
		--print(embeddings[t]:size())
		--print(lstm_c[t-1]:size())
		--print(lstm_h[t-1]:size())
		lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})
		predictions[t] = clones.softmax[t]:forward(lstm_h[t])
		loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
	end

	--backward pass
	local dembeddings = {}
	local dlstm_c = {[opt.seq_length]=dfinalstate_h}
	local dlstm_h = {}
	for t = opt.seq_length, 1, -1 do
		doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])

		if t == opt.seq_length then
			assert(dlstm_h[t] == nil)
			dlstm_h[t] = clones.softmax[t]:backward(lstm_h[t], doutput_t)
		else
			--print('t = '..t)
			--print('dlstm_h[t]') ; print(dlstm_h[t]:size())
			dlstm_h[t]:add(clones.softmax[t]:backward(lstm_h[t], doutput_t))
		end
    
    dembeddings[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.lstm[t]:backward(
        {embeddings[t], lstm_c[t-1], lstm_h[t-1]},
        {dlstm_c[t], dlstm_h[t]}
    ))

		clones.embed[t]:backward(x[{{},t}], dembeddings[t])
	end

	initstate_c:copy(lstm_c[#lstm_c])
	initstate_h:copy(lstm_h[#lstm_h])
	grad_params:clamp(-5, 5)
	return loss, grad_params
end

local losses = {}
local optim_state = {learningRate = 1e-3}
local iterations = opt.max_epochs * loader.nbatches
for i = 1, iterations do
	local _, loss = optim.adam(feval, params, optim_state)
	losses[#losses+1] = loss[1]
	if i % opt.print_every == 0 then
        print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e", i, loss[1], loss[1] / opt.seq_length, grad_params:norm()))
	end
	if i % opt.save_every == 0 then
		torch.save(opt.savefile..'_epoch_'..i..'.t7', protos)
	end
end