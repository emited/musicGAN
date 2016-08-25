local image = require 'image'
local audio = require 'audio'
local signal = require 'signal'

function signal.istft(X, win, hop)
	local S, L = X:size(1), X:size(2)
	local Xc = X:clone():view(S, L/2, -1)
  local x = torch.zeros((S-1)*hop + win)
  framesamp = L/2
  hopsamp = hop
  for n = 1, S do
      i = 1 + (n-1)*hopsamp
      --print(i, i + framesamp - 1)
      x[{{i, i + framesamp - 1}}] = x[{{i, i + framesamp - 1}}]
      	+ signal.complex.real(signal.ifft(Xc[n]))
  end
  return x
end


local DataSet = torch.class('mrnn.DataSet')

function DataSet:__init(opt)
	assert(opt.data_file_path ~= nil)
	assert(opt.sample_save_path ~= nil)
	assert(opt.model_save_path ~= nil)
	assert(opt.seq_length ~= nil)
	assert(opt.batch_size ~= nil)
	self.opt = mrnn.tools.copy(opt)
end


function DataSet:load()
	-- audio file has to be mono
	print('loading '..self.opt.data_file_path..'...')
	local raw, sample_rate = audio.load(self.opt.data_file_path)
	local stop = math.floor(self.opt.data_ratio*raw:size(1))
	return raw:sub(1, stop):squeeze()
end

function DataSet:rescale(data)
	self.min = data:min()
	self.mean = data:mean()
	self.max = data:max()
	return (data-self.mean)/(self.max-self.min)
end

function DataSet:unscale(data)
	return data*(self.max-self.min)+self.min
end

function DataSet:standardize(data)
	self.std = data:std()
	self.mean = data:mean()
	return (data-self.mean)/self.std
end

function DataSet:destandardize(data)
	--launch normalize before
	return data*self.std+self.mean
end


function DataSet:stft(data)
	print('computing stft...')
	local stft = signal.stft(data, self.opt.framesamp, self.opt.hopsamp, self.opt.window_function)
	return stft:view(-1, 1)
end


function DataSet:istft(data)
	print('computing istft...')
	return signal.istft(data, self.opt.framesamp, self.opt.hopsamp)
end


function DataSet:buildBatches(data)
	local D = data:size(1)
	local L = self.opt.seq_length
	local B = self.opt.batch_size
	local I = self.opt.framesamp*2
	local X, Y= {}, {}
	for i = 1, self.opt.nb_batches do
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