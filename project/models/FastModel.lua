local FastModel = torch.class('mrnn.FastModel', 'mrnn.Model')

function FastModel:__init(opt)
	self.opt = mrnn.tools.copy(opt)
end

function FastModel:build()
end

function FastModel:train()
end

function FastModel:sample()
end

