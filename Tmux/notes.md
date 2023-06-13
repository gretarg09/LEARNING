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


# Scriptable tmux commands
* tmux new-session -s development editor  - Creates a session named "development" and names the first window "editor".
* tmux attach -t developlment  - Attaches to a session named "development".
* tmux send-keys -t development '[keys]' C-m  - Sends the keystrokes to the "development" sessions active window or pane. C-m
                                                is equivalent to pressin the ENTER key.
* tmux send-keys -t development:1.1 '[keys]' C-m - Send the keystrokes to the "development" session's first window and first pane, 
                                                   provided the window and pane indexes are set to 1. C-m is equivalent to pressing the 
                                                   ENTER key.
* tmux select-window -t development:1            - Selects the first window of "development", making it the active window.
* tmux split-window -v -p 10 -t development      - Splits the current window in the "development" session vertically, dividing it in half
                                                   horizontally, and sets its height to 10% of the total window size.
* tmux select-layout -t development main-horizontal - sets the layout for the "development" session to main-horizontal.
* tmux source-file [file]                           - Loads the specified tmux configuration file.
* tmux -f app.conf attach                           - Loads the app.conf configuration file and attaches to a session created within the 
                                                      app.conf file.

# Tmuxinator commands

* tmuxinator open [name]  - Opens the configuration file for the project name in the dfault text editor, Creates the configuration if
                            it doesen't exist.
* tmuxinator [name]       - Loads the tmux session for the given project. Creates the session from the contents of the project's
                            configuration file if no session currently exists, or attaches to the session.
* tmuxinator list         - lists all current projects.
* tmuxinator copy [source] [destination] - Copies a project configuration.
* tmuxinator delete [name]  - Deletes the specified project.
* tmuxinator implode        - Deletes all current projects.
* tmuxinator doctor         - Looks for problems with the tmuxinator and system configuration. 
* tmuxinator debug          - Shows the script that tmuxinator will run, helping you figure out what's going wrong.


# MY SETUP

* I switch over to using tmux 2023-06-09. Its simply much easier to manage.
* I found the fzf script [here](https://www.youtube.com/watch?v=BDYaUtOoCP8)
