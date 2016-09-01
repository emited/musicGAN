
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
	self.opt = mrnn.tools.copy(opt)
end

function FastModel:build(X, Y)
	self.X, self.Y = X, Y
	local I = self.opt.framesamp*2
	local H = self.opt.rnn_size
	self.model = nn.Sequencer(
		nn.Sequential()
			:add(nn.FastLSTM(I, H))
			:add(nn.Linear(H, I))
			:add(nn.Tanh())
	)
	self.params, self.grad_params = self.model:getParameters()

	self.feval = function(params_)
		if params_ ~= self.params then
			self.params:copy(params_)
		end
		self.grad_params:zero()
		local _nidx_ = math.random(#X)
		local output = self.model:forward(X[_nidx_])
		local loss = self.criterion:forward(output, Y[_nidx_])
		local delta = self.criterion:backward(output, Y[_nidx_])
		self.model:backward(output, delta)
		return loss, self.grad_params
	end

end


function FastModel:train()
	local _, loss = optim[self.opt.optim_name](self.feval, self.params, self.opt.optim_state)
	return loss[1], self.grad_params:norm()
end


function FastModel:sample()
	prev_c = torch.zeros(1, self.opt.rnn_size)
	prev_h = prev_c:clone()
	prev_inp = torch.zeros(1)
	samples = torch.Tensor(self.opt.sample_length, self.opt.input_size)
	local voice = audio.samplevoice()
	local dmin, dmax = voice:min(), voice:max()
	for i=1, self.opt.sample_length do
		xlua.progress(i, self.opt.sample_length)
		local next_c, next_h = unpack(self.protos.lstm:forward({prev_inp, prev_c, prev_h}))
		prev_c:copy(next_c)
		prev_h:copy(next_h)
		local next_inp = self.protos.pred:forward(prev_h)
		prev_inp:copy(next_inp)
		samples[i] = prev_inp*(dmax-dmin)+dmin
	end
	return samples
end

