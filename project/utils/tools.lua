
local tools = torch.class('mrnn.tools')

function tools.max(table)
	local max, argmax = -math.huge, -1
	for k, v in pairs(table) do
		if max < v then
			max = v
			argmax = k
		end
	end
	return max, argmax
end


function tools.argmax(table)
	local max, argmax = self:max(table)
	return argmax
end


function copy(obj, seen)
	if type(obj) ~= 'table' then return obj end
	if seen and seen[obj] then return seen[obj] end
	local s = seen or {}
	local res = setmetatable({}, getmetatable(obj))
	s[obj] = res
	for k, v in pairs(obj) do res[copy(k, s)] = copy(v, s) end
	return res
end


function tools.copy(table, deep)
	local deep = deep or true
	if deep then
		return copy(table, {})
	else
		local new_table = {}
		for k, v in pairs(table) do
			new_table[k] = v
		end
		return new_table
	end
end

return tools