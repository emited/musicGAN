require 'nn'
require 'nngraph'
require 'optim'
require 'audio'
require 'gnuplot'
require 'LSTM_test'
local signal = require 'signal'
local model_utils = require 'model_utils'
local LSTM = require 'LSTM'

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

------------------------------------------------------
-------------------- PARAMETERS ----------------------
------------------------------------------------------

--setting up parameters
opt = {
	---data
	data_path = '../../../data/voice.mp3',
	savefile = 'lstm',
	seq_length = 200,
	batch_size = 128,

	--model
	max_epochs = 10,
	seed = 122,
	num_layers = 1,
	rnn_size = 256,
	input_size = 1,
	evaluate_every = 1,
	sample_every = 50,
	print_every = 1,
	save_every = 50,
	optim_name = 'adam',
	optim_state = {learningRate = 1e-2,},
	sample_length = 50000,
}

------------------------------------------------------
------------------- MODEL SAMPLE ---------------------
------------------------------------------------------
function sample(opt, protos, suffix)
	print('sampling...')
	prev_c = torch.zeros(1, opt.rnn_size)
	prev_h = prev_c:clone()
	prev_inp = torch.zeros(1)
	samples = torch.Tensor(opt.sample_length, opt.input_size)
	local voice = audio.samplevoice()
	local dmin, dmax = voice:min(), voice:max()
	for i=1, opt.sample_length do
		xlua.progress(i, opt.sample_length)
		local next_c, next_h = unpack(protos.lstm:forward({prev_inp, prev_c, prev_h}))
		prev_c:copy(next_c)
		prev_h:copy(next_h)
		local next_inp = protos.pred:forward(prev_h)
		prev_inp:copy(next_inp)
		samples[i] = prev_inp*(dmax-dmin)+dmin
	end
	torch.save('sample'..suffix, samples)
	audio.save('sample'..suffix..'.mp3', samples, 11025)
	--audio.save('sample.mp3', samples, 11025)
end



------------------------------------------------------
----------------------- DATA -------------------------
------------------------------------------------------

--loading input data
-- sine data
local voice = audio.samplevoice()
local dmin, dmax = voice:min(), voice:max()
local x_sin = torch.linspace(0, 10000, 700*1024)
local tmp_sin = x_sin*math.pi*2
local y_sin = torch.sin(tmp_sin)
local data = y_sin
local test_data = (data*(dmax-dmin)+dmin):sub(1, 100000):view(-1, 1)
audio.save('train_sample.mp3', test_data, 11025)

-- samplevoice data
--local data = audio.load(opt.data_path):view(-1)
--local dmin = data:min()
--local dmax = data:max()
--data = (data-dmin)/(dmax-dmin)

local ydata = data:clone()
ydata:sub(1, -2):copy(data:sub(2, -1))
ydata[-1] = data[1]
local x_batches = data:view(opt.batch_size, -1):split(opt.seq_length, 2)
local y_batches = ydata:view(opt.batch_size, -1):split(opt.seq_length, 2)
if x_batches[#x_batches]:size(2) ~= opt.seq_length then
	table.remove(x_batches, #x_batches)
	table.remove(y_batches, #y_batches)
end
assert(#x_batches == #y_batches)
local nbatches = #x_batches

------------------------------------------------------
-------------- MODEL INIT and TRAIN ------------------
------------------------------------------------------

torch.manualSeed(opt.seed)

local protos = {}
protos.lstm = LSTM.lstm(opt)
--protos.lstm = nn.LSTM(opt.input_size, opt.rnn_size)
protos.pred = nn.Sequential():add(nn.Linear(opt.rnn_size, 1))
protos.criterion = nn.MSECriterion()

local params, grad_params = model_utils.combine_all_parameters(protos.lstm, protos.pred)
params:uniform(-0.008, 0.008)

local clones = {}
for name, proto in pairs(protos) do
	print('cloning '..name)
	clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.paramters)
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
	local _nidx_ = math.random(nbatches)
	local x, y = x_batches[_nidx_], y_batches[_nidx_]

	local lstm_c = {[0]=initstate_c}
	local lstm_h = {[0]=initstate_h}
	local predictions = {}
	local loss = 0

	for t = 1, opt.seq_length do
		local input = x[{{}, t}]:clone():view(opt.batch_size, opt.input_size)
		local target = y[{{}, t}]:clone():view(opt.batch_size, opt.input_size)
		lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{input, lstm_c[t-1], lstm_h[t-1]})
		predictions[t] = clones.pred[t]:forward(lstm_h[t])
		loss = loss + clones.criterion[t]:forward(predictions[t], target)
	end

	--backward pass
	local dlstm_c = {[opt.seq_length]=dfinalstate_h}
	local dlstm_h = {}
	for t = opt.seq_length, 1, -1 do
		local input = x[{{}, t}]:clone():view(opt.batch_size, opt.input_size)
		local target = y[{{}, t}]:clone():view(opt.batch_size, opt.input_size)
		doutput_t = clones.criterion[t]:backward(predictions[t], target)
		if t == opt.seq_length then
			assert(dlstm_h[t] == nil)
			dlstm_h[t] = clones.pred[t]:backward(lstm_h[t], doutput_t)
		else
			dlstm_h[t]:add(clones.pred[t]:backward(lstm_h[t], doutput_t))
		end

		dinput, dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.lstm[t]:backward({input, lstm_c[t-1], lstm_h[t-1]},
				{dlstm_c[t], dlstm_h[t]}))
	end
	initstate_c:copy(lstm_c[#lstm_c])
	initstate_h:copy(lstm_h[#lstm_h])
	grad_params:clamp(-5, 5)
	return loss, grad_params
end


local losses = {}
local iterations = opt.max_epochs * nbatches
for i = 1, iterations do
	xlua.progress(i, iterations)
	local _, loss = optim[opt.optim_name](feval, params, opt.optim_state)
	losses[#losses+1] = loss[1]
	if i % opt.print_every == 0 then
		gnuplot.plot(torch.Tensor(losses))
    print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e", i, loss[1], loss[1] / opt.seq_length, grad_params:norm()))
	end
	if i % opt.save_every == 0 then
		torch.save(opt.savefile..'_epoch_'..i..'.t7', protos)
	end
	if i % opt.sample_every == 0 then
		sample(opt, protos, i)
	end
end



