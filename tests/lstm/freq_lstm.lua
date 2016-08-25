require 'nn'
require 'nngraph'
require 'optim'
require 'audio'
require 'gnuplot'
require 'LSTM_test'
local signal = require 'signal'
local model_utils = require 'model_utils'
local LSTM = require 'LSTM'

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

--data
local data = {}
data.opt = opt




local net = {}
net.data = 1

function net:train()
	print(self.data)
	print('training...')
end

net:train()
