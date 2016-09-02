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
	opt = mrnn.tools.copy(opt)
end


function DataSet:load()
	-- audio file has to be mono
	print('loading '..opt.data_file_path..'...')
	local raw, rate = audio.load(opt.data_file_path)
	print('sample rate  = '..rate)
	self.sample_rate = rate
	local stop = math.floor(opt.data_ratio*raw:size(1))
	return raw:sub(1, stop):squeeze()
end

function DataSet:rescale(data)
	self.min = data:min()
	self.mean = data:mean()
	self.max = data:max()
	return (data-self.mean)/(self.max-self.min)
end

function DataSet:normalize(data)
	local method = opt.normalize
	if method == 'standardize' then
		self.std = data:std()
		self.mean = data:mean()
		return (data-self.mean)/self.std
	elseif method == 'squash' then
		self.min = data:min()
		self.max = data:max()
		self.mean = data:mean()
		return (data-self.mean)/(self.max-self.min)
	elseif method == 'scale' then
		self.min = data:min()
		self.max = data:max()
		return (data-self.min)/(self.max-self.min)
	end
end

function DataSet:unnormalize(data)
	local method = opt.normalize
	if method == 'standardize' then
			return data*self.std+self.mean
	elseif method == 'squash' then
		return data*(self.max-self.min)+self.mean
	elseif method == 'scale' then
		return data*(self.max-self.min)+self.min
	end
end

function DataSet:stft(data)
	print('computing stft...')
	local stft = signal.stft(data, opt.framesamp, opt.hopsamp, opt.window_function)
	return stft:view(stft:size(1), -1)
end

function DataSet:istft(data)
	print('computing istft...')
	return signal.istft(data, opt.framesamp, opt.hopsamp)
end

function DataSet:save(filename, data)
	local filepath = opt.sample_save_path..filename
	audio.save(filepath, data:view(-1,1), self.sample_rate)
end


function DataSet:display(data)
	print('displaying data...')
	local os = require 'os'
	torch.save('display.t7', data)
	local command = '\'require(\"image\") disp = torch.load(\"display.t7\") image.display(disp) \''
	os.execute(	'echo '..command..' >  display.lua')
	os.execute('qlua display.lua')
	os.execute('rm display.lua')
	os.execute('rm display.t7')
end
