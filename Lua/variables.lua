
-- This is a comment
name = 'Derek'
io.write('size of string ', #name, '\n')


-- Strings
longString = [[
I am a very very long
string that goes on forever
]]

longString = longString .. name

io.write(longString, "\n")

-- BooleanllongString
isAbleToDrive = true
io.write(type(isAbleToDrive), "\n")

-- All variables in lua get the value nil as default.
io.write(type(madeUpVar), "\n") -- 
