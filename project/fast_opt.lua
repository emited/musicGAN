
local opt = {

	--print & plotting
	evaluate_every = 1,
	print_every = 1,
	sample_every = 50,
	save_every = 10000,
	plot_loss = true,
	plot_data = false,
	
	---data
	data_file_path = '../data/test_sine.mp3',
	model_save_path = 'saved/models/',
	sample_save_path = 'saved/samples/',
	normalize = 'squash',
	data_ratio = 1,
	seq_length = 64,
	batch_size = 64,
	nb_batches = 32,
	framesamp = 1024,
	hopsamp = 512,
	window_function = 'hann',

	--model
	seed = 126,
	model_name = 'FastModel',
	max_epochs = 10000,
	num_layers = 2,
	rnn_size = 200,
	optim_name = 'adam',
	optim_state = {learningRate = 1e-2,},
	sample_length = 1000,
	uniform = 0.08,
	remember = 'neither', --can be train, neither, eval, or both
	batch_norm = false,
	use_nngraph = true,
	cuda = false,
}

return opt
