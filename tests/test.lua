require 'nn'
require 'nngraph'
require 'optim'
require 'audio'
require 'LSTM'

------------------------------------------------------
-------------------- PARAMETERS ----------------------
------------------------------------------------------

--setting up parameters
opt = {
	---data
	data_path = '../../data/voice.mp3',
	input_dim = 20,
	target_dim = 5,
	batch_dim = 1,
	fold_gap = 5,

	--model
	num_layers = 1,
	rnn_size = 5,
	evaluate_every = 1,
	max_epoch = 10000,
	optim_name = 'adam',
	optim_parameters = {
		learningRate = 1e-3,
	}
}

------------------------------------------------------
----------------------- DATA -------------------------
------------------------------------------------------

--loading input data
data = audio.load(opt.data_path):view(-1)
data = data[{{1, 401}}]
data_size = data:size(1)
print('data size = '..data_size)

-- making folds
folds = {}
local fold_dim = opt.input_dim + opt.target_dim
local k = 1
for i = 1, data_size-fold_dim+1, opt.fold_gap do	
	folds[k] = {}
	for j = 1, fold_dim do
		folds[k][j] = data[i+j-1]
	end
	k = k + 1
end
print('number of folds = '..(k-1))

--making batches
batches = {}
batches.full = {}
local k = 1
local perm = torch.randperm(#folds)
for i = 1, #folds-opt.batch_dim, opt.batch_dim do
	batches.full[k] = {}
	for j = 1, opt.batch_dim do
		batches.full[k][j] = folds[perm[i+j-1]]
	end
	k = k + 1
end
batches.full = torch.Tensor(batches.full)
batches.dim = batches.full:size(1)
batches.input = batches.full[{{},{},{1, opt.input_dim}}]
batches.target = batches.full[{{},{},{opt.input_dim+1, opt.target_dim+opt.input_dim}}]
print('number of batches = '..batches.dim)


------------------------------------------------------
--------------------- MODEL --------------------------
------------------------------------------------------

c0 = torch.zeros(batches.dim, opt.target_dim)
h0 = torch.zeros(batches.dim, opt.target_dim)
x = batches


--initializing network
net = nn.Sequential()
rnns = {}
for i = 1, opt.num_layers do
	local prev_dim = opt.rnn_size
	if i == 1 then prev_dim = opt.input_dim end
	rnn = nn.LSTM(prev_dim, opt.target_dim)
	rnn.remember_states = true
	rnns[#rnns] = rnn
	net:add(rnn)
end
--view1 = nn.View(1, 1, -1):setNumInputDims(3)
--view2 = nn.View(1, -1):setNumInputDims(2)
--net:add(view1)
--net:add(nn.Linear(opt.target_dim, opt.rnn_size))
--net:add(view2)


--lstm = nn.LSTM(opt.input_dim, opt.target_dim)
criterion = nn.MSECriterion()
params, gradParams = net:getParameters()

--preparing train
feval = function(params_new)
	if params ~= params_new then params:copy(params_new) end
	h = net:forward(x.input[1])
	print(h:size())
	loss_x = criterion:forward(h, x.target)
	grad_h = criterion:backward(h, x.target)
	--grad_c, grad_h, grad_x = unpack(lstm:backward({c0, h0, x.input}, grad_h))
	return loss_x, gradParams
end

--training
local epoch = 0
for i = 1, opt.max_epoch do
	local loss = 0
	for j = 1, batches.dim do
		_, fs = optim[opt.optim_name](feval, params, opt.optim_parameters)
		loss = loss + fs[1]
	end
	if epoch%opt.evaluate_every == 0 then
		print('loss = '..loss)
	end
	epoch = epoch + 1
end