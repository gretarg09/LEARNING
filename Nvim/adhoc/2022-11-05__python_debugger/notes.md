# My notes

## **[Commands]**
* Use ```lua :lua require('dap').continue``` to sart debugging
* See ```lua help dap-mappings``` and  ```lua help dap-api```
* Use ```:lua require('dap-python').test_method()``` to debug the closest method above
  the cursor.

## Current status
The debugger works for this simple example. If you run ```:lua require('dap-python').test_method()```
when you are in the ```python test_adding_two_numbers``` then the debugger ui will be rendered and 
the debuggin session will start on the first breakpoint. 


## Resources
* [article 1](https://alpha2phi.medium.com/neovim-lsp-and-dap-using-lua-3fb24610ac9f)
* [DAP - debug adapter protocol](https://github.com/mfussenegger/nvim-dap)
* [nvim dap python](https://github.com/mfussenegger/nvim-dap-python)
* [debugpy](https://github.com/mfussenegger/nvim-dap-python)

