require 'nn'
require 'nngraph'
require 'optim'
require 'audio'
require 'LSTM'

--------------------------------------------
--------------- PARAMETERS -----------------
--------------------------------------------

--setting up parameters
opt = {
	data_path = '../../data/voice.mp3',
	input_dim = 7,
	target_dim = 3,
	hidden_dim = 10,
	batch_dim = 9,
	fold_gap = 2,
}

--------------------------------------------
------------------ DATA --------------------
--------------------------------------------

--loading input data
data = audio.load(opt.data_path):view(-1)
data = data[{{1, 41}}]
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


--------------------------------------------
----------------- MODEL --------------------
--------------------------------------------

--initializing LSTM network
x = batches
c0 = torch.zeros(batches.dim, opt.hidden_dim)
h0 = torch.zeros(batches.dim, opt.hidden_dim)

lstm = nn.LSTM(opt.input_dim, opt.hidden_dim)
criterion = nn.MSECriterion()

-- forward and backward
h = lstm:forward(x.input)
print(h)
loss_x = criterion:forward(h, x.target)
grad_h = criterion:backward(h, x.target)
grad_c0, grad_h0, grad_x = unpack(lstm:backward(x.input, grad_h))