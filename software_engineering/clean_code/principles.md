# My rules

**[Functions]**
* Be small.
* Do one thing.
* Contain code with the same level of abstraction.
* Have fewer than 4 arguments (~ I am not sure about this).
* Have not duplication.
* Use descriptive name.
* Try to maximize the cohasion of the code (things that belong together should stick together). 

**[Architecture]**

* Organize the code into modules and create an interface for the module. 
* choose small scripts over large scripts.
* Write Dry code within modules but try to avoid creating coupling between modules. The coupling is more important, even if it 
  means that you need to doublicate samll amount of code. Use your best judgement here. 

## Resources

[python clean code: 6 best practices to make your python functions more readable](https://towardsdatascience.com/python-clean-code-6-best-practices-to-make-your-python-functions-more-readable-7ea4c6171d60)
