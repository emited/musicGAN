
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
	assert(opt.batch_norm ~= nil)
	assert(opt.use_nngraph ~= nil)
	assert(opt.uniform ~= nil)
	assert(opt.remember ~= nil)
	assert(opt.cuda ~= nil)
end

function FastModel:build(data)
	self.X, self.Y = self:buildBatches(data)
	self.model = self:buildModule()
	self.seq_model = nn.Sequencer(self.model)
	self.seq_model:remember(opt.remember)
	self.params, self.grad_params = self.seq_model:getParameters()
	self.params:uniform(-opt.uniform, opt.uniform)

	self.crit = nn.MSECriterion()
	self.seq_crit = nn.SequencerCriterion(self.crit)

	self.params:uniform(opt.uniform)
	self.optim_state = mrnn.tools.copy(opt.optim_state)

	self.feval = function(params_)
		if params_ ~= self.params then
			self.params:copy(params_)
		end
		self.grad_params:zero()
		local _nidx_ = math.random(#self.X)
		local x, y = self.X[_nidx_], self.Y[_nidx_]
		self.last_x = x[-1][-1]
		local output = self.seq_model:forward(x)
		local loss = self.seq_crit:forward(output, y)
		local delta = self.seq_crit:backward(output, y)
		self.seq_model:backward(output, delta)
		return loss, self.grad_params
	end

end

function FastModel:buildModule()
	local I = opt.framesamp*2
	local H = opt.rnn_size
	local model = nn.Sequential()
	nn.FastLSTM.bn = opt.batch_norm
	nn.FastLSTM.usenngraph = opt.usenngraph
	model:add(nn.FastLSTM(I, H))
	for i = 2, opt.num_layers do model:add(nn.FastLSTM(H, H))	end
	model:add(nn.Linear(H, I))
	model:add(nn.Tanh())
	return model
end

function FastModel:buildBatches(data)
	-- returns table of table of tensors
	-- of size N*I
	print('building batches...')
	local X, Y = {}, {}
	local D = data:size(1)
	local I = data:size(2)
	local B = opt.nb_batches
	local N = opt.batch_size
	local T = opt.seq_length
	for i = 1, B do
		X[i] = torch.Tensor(T, N, I)
		Y[i] = torch.Tensor(T, N, I)
		local start = {}
		for j = 1, T do
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
	self.seq_model = self.seq_model:cuda()
	self.seq_crit = self.seq_crit:cuda()
end

function FastModel:train()
	self.seq_model:training()
	local _, loss = optim[opt.optim_name](self.feval, self.params, self.optim_state)
	return loss[1], self.grad_params:norm()
end


function FastModel:sample()
	self.seq_model:evaluate()
	self.model:evaluate()
	local T = opt.sample_length
	local I = opt.framesamp*2
	local samples = torch.Tensor(T+1, I)
	for i = 2, T do
		xlua.progress(i, T)
		if i == 1 then sample[1] = self.last_x end
		samples[i] = self.model:forward(samples[i-1])
	end
	return samples
end

function FastModel:save(filename)
	print('saving FastModel...')
	local filepath = opt.data_file_path..filename
	print('not implemented yet.')
end