-- Relational Operators : > < >= <= == ~=

-- Logical Operators : and or not

age = 17

if age < 16 then
    io.write("You can go to school", "\n")
    local localVar = 10
elseif (age <= 16) and (age < 18) then
    io.write("You can drive", "\n")
else
    io.write("You can vote", "\n")
end


age = 13

        if (age < 14) or (age > 67) then io.write("You shouldn't work") end
