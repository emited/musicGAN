require 'mrnn'
require 'gnuplot'
local opt = require(arg[1])
print(opt)

torch.manualSeed(opt.seed)

--data
local dataset = mrnn.DataSet(opt)
local sig = dataset:load()
local nsig = dataset:normalize(sig)
local stft = dataset:stft(nsig)

--model
local model = mrnn[opt.model_name](opt)
if opt.cuda then model:toCuda() end
model:build(stft)

local losses = {}
local max_steps = opt.max_epochs*opt.nb_batches
for i = 1, max_steps do
	xlua.progress(i, max_steps)

	--training
	local loss, grad_norm = model:train()
	losses[#losses+1] = loss

	--printing and plotting	
	if i % opt.print_every == 0 then
		if opt.plot_loss then gnuplot.plot(torch.Tensor(losses)) end
    print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f, grad_norm = %6.8f", i, loss, loss / opt.seq_length, grad_norm))
	end

	--sampling
	if i % opt.sample_every == 0 then
		local t = os.time()
    t = os.date('%d_%b_%Y_%H:%M:%S',t)
		local output = model:sample()
		local sig = dataset:istft(output)
		local nsig = dataset:destandardize(sig)
		audio.save('sample'..t..'.mp3', nsig, opt.sample_rate)
	end

	--saving model
	--if i % opt.save_every == 0 then
	--	torch.save(opt.savefile..'_epoch_'..i..'.t7', protos)
	--end

end