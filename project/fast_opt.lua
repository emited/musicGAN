
local opt = {
	model_name = 'FastModel',
	max_epochs = 100,
	seed = 122,
	evaluate_every = 1,
	print_every = 1,
	sample_every = 200,
	save_every = 10000,
	plot_loss = true,
	sample_rate = 11025,
	
	---data
	data_file_path = '../data/classics_11025.mp3',
	model_save_path = 'saved/',
	sample_save_path = 'saved/',
	normalize = 'squash',
	data_ratio = 0.01,
	seq_length = 17,
	batch_size = 27,
	nb_batches = 10,
	framesamp = 1024,
	hopsamp = 512,
	window_function = 'hann',

	--model
	num_layers = 1,
	rnn_size = 2048,
	optim_name = 'adam',
	optim_state = {learningRate = 1e-2,},
	sample_length = 10000,
	batch_norm = false,
	cuda = true,
}

return opt
