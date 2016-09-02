
require 'nn'
require 'nngraph'
require 'optim'

local SimpleModel = torch.class('mrnn.SimpleModel', 'mrnn.Model')

function SimpleModel:__init(opt)
	assert(opt.num_layers ~= nil)
	assert(opt.seq_length ~= nil)
	assert(opt.rnn_size ~= nil)
	assert(opt.optim_name ~= nil)
	assert(opt.optim_state ~= nil)
	assert(opt.sample_length ~= nil)
end


function SimpleModel:build(data)
	self.X, self.Y = self:buildBatches(data)
	self.protos = {}
	opt.input_size = opt.framesamp*2
	self.protos.lstm = mrnn.LSTM(opt):build()
	self.protos.pred = nn.Sequential():add(nn.Linear(opt.rnn_size, opt.input_size))
	self.protos.criterion = nn.MSECriterion()
	self.params, self.grad_params = mrnn.model_utils.combine_all_parameters(self.protos.lstm, self.protos.pred)
	self.params:uniform(-0.008, 0.008)
	self.clones = {}
	for name, proto in pairs(self.protos) do
		print('cloning '..name)
		self.clones[name] = mrnn.model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
	end
	self.initstate_c = torch.zeros(opt.batch_size, opt.rnn_size)
	self.initstate_h = self.initstate_c:clone()
	self.dfinalstate_c = self.initstate_c:clone()
	self.dfinalstate_h = self.initstate_c:clone()

	self.feval = function(params_)
		if params_ ~= self.params then
			self.params:copy(params_)
		end
		self.grad_params:zero()

		--forward pass
		local _nidx_ = math.random(#X)
		local x, y = self.X[_nidx_], self.Y[_nidx_]
		local I = opt.framesamp*2
		local B = opt.batch_size
		local L = opt.seq_length
		local lstm_c = {[0]=self.initstate_c}
		local lstm_h = {[0]=self.initstate_h}
		local predictions = {}
		local loss = 0
		for t = 1, opt.seq_length do
			local input = x[{{}, t}]:clone()
			local target = y[{{}, t}]:clone()
			lstm_c[t], lstm_h[t] = unpack(self.clones.lstm[t]:forward{input, lstm_c[t-1], lstm_h[t-1]})
			predictions[t] = self.clones.pred[t]:forward(lstm_h[t])
			loss = loss + self.clones.criterion[t]:forward(predictions[t], target)
		end
		print('predictions')
		print(predictions)
		--backward pass
		local dlstm_c = {[L]=self.dfinalstate_h}
		local dlstm_h = {}
		for t = L, 1, -1 do
			local input = x[{{}, t}]:clone():view(B, I)
			local target = y[{{}, t}]:clone():view(B, I)
			doutput_t = self.clones.criterion[t]:backward(predictions[t], target)
			if t == L then
				assert(dlstm_h[t] == nil)
				dlstm_h[t] = self.clones.pred[t]:backward(lstm_h[t], doutput_t)
			else
				dlstm_h[t]:add(self.clones.pred[t]:backward(lstm_h[t], doutput_t))
			end
			dinput, dlstm_c[t-1], dlstm_h[t-1] = unpack(self.clones.lstm[t]:backward({input, lstm_c[t-1], lstm_h[t-1]},
					{dlstm_c[t], dlstm_h[t]}))
		end
		self.initstate_c:copy(lstm_c[#lstm_c])
		self.initstate_h:copy(lstm_h[#lstm_h])
		self.grad_params:clamp(-5, 5)
		return loss, self.grad_params
	end

end


function SimpleModel:train()
	local _, loss = optim[opt.optim_name](self.feval, self.params, opt.optim_state)
	return loss[1], self.grad_params:norm()
end


function SimpleModel:buildBatches(data)
	local D = data:size(1)
	local L = opt.seq_length
	local B = opt.batch_size
	local I = opt.framesamp*2
	local X, Y= {}, {}
	for i = 1, opt.nb_batches do
		X[i] = torch.Tensor(B, L, I)
		Y[i] = torch.Tensor(B, L, I)
		for j = 1, B do
			local _nidx_ = math.random(D-L+1)
			for k = 1, L do
				X[i][j][k] = data[_nidx_+j-1]
				Y[i][j][k] = data[_nidx_+j]
			end
		end
	end
	return X, Y
end


function SimpleModel:sample()
	prev_c = torch.zeros(1, opt.rnn_size)
	prev_h = prev_c:clone()
	prev_inp = torch.zeros(1)
	samples = torch.Tensor(opt.sample_length, opt.input_size)
	local voice = audio.samplevoice()
	local dmin, dmax = voice:min(), voice:max()
	for i=1, opt.sample_length do
		xlua.progress(i, opt.sample_length)
		local next_c, next_h = unpack(self.protos.lstm:forward({prev_inp, prev_c, prev_h}))
		prev_c:copy(next_c)
		prev_h:copy(next_h)
		local next_inp = self.protos.pred:forward(prev_h)
		prev_inp:copy(next_inp)
		samples[i] = prev_inp*(dmax-dmin)+dmin
	end
	return samples
end

