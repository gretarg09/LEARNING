io.write("5 + 3 = ", 5+3, "\n")
io.write("5 - 3 = ", 5-3, "\n")
io.write("5 * 3 = ", 5*3, "\n")
io.write("5 / 3 = ", 5/3, "\n")
io.write("5.2 % 3 = ", 5%3, "\n")


-- Math functions: floor,  ceil, max, min, sin , cos, asin, acos, exp
-- log, log10, pow, sqrt, random
--
io.write("floor(2.345) : ", math.floor(2.345), "\n")
io.write("ceil(2.345) : ", math.ceil(2.345), "\n")
io.write("max(2.345) : ", math.max(2.345), "\n")
io.write("min(2.345) : ", math.min(2.345), "\n")
io.write("sqrt(2.345) : ", math.sqrt(2.345), "\n")


-- Random numbers
math.randomseed(os.time()) -- change the seed
io.write("math.random() : ", math.random(), "\n")
io.write("math.random(10) : ", math.random(10), "\n")
io.write("math.random(5, 100) : ", math.random(5, 100), "\n")

-- String formatting
print(string.format("pi = %.10f", math.pi))
