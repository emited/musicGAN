
local opt = {
	---data
	data_file_path = '../data/classics_11025.mp3',
	model_save_path = 'saved/',
	sample_save_path = 'saved/',
	data_ratio = 0.01,
	seq_length = 17,
	batch_size = 27,
	nb_batches = 13,
	framesamp = 1024,
	hopsamp = 512,
	window_function = 'hann',
	max_epochs = 10,
	seed = 122,
	evaluate_every = 1,
	print_every = 1,
	sample_every = 200,
	save_every = 10000,
	sample_rate = 11025,	
	--model
	num_layers = 1,
	rnn_size = 256,
	optim_name = 'adam',
	optim_state = {learningRate = 1e-4,},
	sample_length = 10000,
}

return opt
