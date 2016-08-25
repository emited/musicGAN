require 'nn'
require 'nngraph'

local LSTM = torch.class('mrnn.LSTM')

function LSTM:__init(opt)
    self.opt = mrnn.tools.copy(opt)
end


-- Creates one timestep of one LSTM
function LSTM:build()
    local x = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    function new_input_sum()
        -- transforms input
        local i2h            = nn.Linear(self.opt.input_size, self.opt.rnn_size)(x)
        -- transforms previous timestep's output
        local h2h            = nn.Linear(self.opt.rnn_size, self.opt.rnn_size)(prev_h)
        return nn.CAddTable()({i2h, h2h})
    end

    local in_gate          = nn.Sigmoid()(new_input_sum())
    local forget_gate      = nn.Sigmoid()(new_input_sum())
    local out_gate         = nn.Sigmoid()(new_input_sum())
    local in_transform     = nn.Tanh()(new_input_sum())

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    return nn.gModule({x, prev_c, prev_h}, {next_c, next_h})
end

return LSTM
