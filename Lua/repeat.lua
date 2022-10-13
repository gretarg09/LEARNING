repeat
   io.write('Enter your guess: ')
   local guess = io.read()
until tonumber(guess) == 15
