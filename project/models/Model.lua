
local Model = torch.class('mrnn.Model')

function Model:__init(opt)
	self.opt = mrnn.tools.copy(opt)
end

function Model:train()
	return nil
end

function Model:sample()
	return nil
end