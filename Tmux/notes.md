# Commands

´´´tmux ´´´
´´´tmux ls ´´´
´´´tmux new -s <session-name>´´´
´´´tmux attach -t <session-name> ´´´
´´´tmux kill-session -t <session-name> ´´´
´´´tmux new -s window -n shell ´´´ - using the n-flag will tell tmux to name the first window.
´´´ ´´´
´´´ ´´´
´´´ ´´´

# Keybindings
* PREFIX + d : detach
* PREFIX + t : time
* PREFIX + c : new window
* PREFIX + , : renaming a window 
* PREFIX + n : move to next window.
* PREFIX + p : move to previous window.
* PREFIX + w : display a visual menu of your windows.
* PREFIX + ? : List of all predefined tmux keybindings and associated commands these trigger.


# Keybindings Default Commands for Sessions, Windows, and Panes

* PREFIX + d : Detaches from the session, leaving the session running in the
              background.
* PREFIX + : : Enters Command mode.
* PREFIX + c : Creates a new window from within an existing tmux session.
               Shortcut for new-window .
* PREFIX + n : Moves to the next window.
* PREFIX + p : Moves to the previous window.
* PREFIX + 0 … 9 : Selects windows by number.
* PREFIX + w : Displays a selectable list of windows in the current session.
 PREFIX + f : Searches for a window that contains the text you specify.
               Displays a selectable list of windows containing that text in the
               current session.
* PREFIX + , : Displays a prompt to rename a window.
* PREFIX + & : Closes the current window after prompting for confirmation.
* PREFIX + % : Divides the current window in half vertically.
* PREFIX + " : Divides the current window in half horizontally.
* PREFIX + o : Cycles through open panes.
* PREFIX + q : Momentarily displays pane numbers in each pane.
* PREFIX + x : Closes the current pane after prompting for confirmation.
* PREFIX + S : PACE Cycles through the various pane layouts.
* PREFIX + SPACEBAR: Cycle through the default pane layouts.


# Creating Sessions using command mode. 

* ´´´tmux new-session´´´  :  Creates a new session without a name. Can be
                            shortened to tmux new or simply tmux .
* ´´´tmux new -s development´´´ : Creates a new session called “development.”
* ´´´tmux new -s development -n editor´´´ :  Creates a session named “development” and
                                             names the first window “editor.”
* ´´´tmux attach -t development´´´ :  Attaches to a session named “development.”


# Restart tmux after config changes

Enter command mode by pressing ´´´PREFIX :´´´
Then enter ´´´source-file ~/.tmux.conf´´´

# Bind a key to deries of commands

You can bind a series of commands by separating the commands with the 
\; character combination.
