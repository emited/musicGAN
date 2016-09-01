
require 'nn'
require 'rnn'
require 'nngraph'
require 'optim'

local FastModel = torch.class('mrnn.FastModel', 'mrnn.Model')

function FastModel:__init(opt)
	assert(opt.num_layers ~= nil)
	assert(opt.seq_length ~= nil)
	assert(opt.rnn_size ~= nil)
	assert(opt.optim_name ~= nil)
	assert(opt.optim_state ~= nil)
	assert(opt.sample_length ~= nil)
	assert(opt.cuda ~= nil)
	self.opt = mrnn.tools.copy(opt)
end

function FastModel:build(data)
	self.X, self.Y = self:buildBatches(data)
	self.model = self:buildModule()
	self.criterion = nn.SequencerCriterion(nn.MSECriterion())
	self.params, self.grad_params = self.model:getParameters()

	self.feval = function(params_)
		if params_ ~= self.params then
			self.params:copy(params_)
		end
		self.grad_params:zero()
		local _nidx_ = math.random(#self.X)
		local x, y = self.X[_nidx_], self.Y[_nidx_]
		if self.opt.cuda then
			for i = 1, #x do
				x[i] = x[i]:cuda()
				y[i] = y[i]:cuda()
			end
		end
		local output = self.model:forward(x)
		local loss = self.criterion:forward(output, y)
		local delta = self.criterion:backward(output, y)
		self.model:backward(output, delta)
		return loss, self.grad_params
	end

end

function FastModel:buildModule()
	local I = self.opt.framesamp*2
	local H = self.opt.rnn_size
	local mod = nn.Sequential()
	local lstm = nn.FastLSTM(I, H)
	lstm.bn = self.opt.batch_norm
	for i = 1, self.opt.num_layers do mod:add(lstm) end
	mod:add(nn.Linear(H, I))
	mod:add(nn.Tanh())
	return nn.Sequencer(mod)
end

function FastModel:buildBatches(data)
	-- returns table of table of tensors
	-- of size N*I
	local X, Y = {}, {}
	local D = data:size(1)
	local I = data:size(2)
	local B = self.opt.nb_batches
	local N = self.opt.batch_size
	local T = self.opt.seq_length
	for i = 1, B do
		X[i], Y[i] = {}, {}
		local start = {}
		for j = 1, T do
			X[i][j] = torch.Tensor(N, I)
			Y[i][j] = torch.Tensor(N, I)
			for k = 1, N do
				start[k] = start[k] or math.random(D-T-1)
				X[i][j][k] = data[start[k]+j-1]
				Y[i][j][k] = data[start[k]+j]
			end
		end
	end
	return X, Y
end

function FastModel:toCuda()
	cunn = require 'cunn'
	cudnn = require 'cudnn'
	self.model = self.model:cuda()
	self.criterion = self.criterion:cuda()
end

function FastModel:train()
	self.model:train()
	local _, loss = optim[self.opt.optim_name](self.feval, self.params, self.opt.optim_state)
	return loss[1], self.grad_params:norm()
end


function FastModel:sample()
	local T = self.opt.sample_length
	local I = self.opt.framesamp*2
	self.module:evaluate()
	local samples = torch.Tensor(T, I)
	for i = 1, T do
		xlua.progress(i, T)
		--samples[i] = 
	end
	return samples
end

