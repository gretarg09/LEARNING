
[How to write a Git Commit Message](https://cbea.ms/git-commit/)
[Customizing Git - Git configuration](https://git-scm.com/book/en/v2/Customizing-Git-Git-Configuration)
[The pro git book](https://git-scm.com/book/en/v2)


# Cheatsheet
## git log

```git log```
```git log --oneline```  - prints out just the subject line.
```git shortlog```  - groups commit by user, again showing just the subject line.


# Squashing git commits
[A short and precise video about git commit squashing](https://www.google.com/search?q=squashing%20commits&source=lnms&tbm=vid&sa=X&ved=2ahUKEwjI4pODy7z8AhUSX_EDHVOwCDEQ_AUoAXoECAEQAw&biw=2749&bih=1578&dpr=1.05#fpstate=ive&vld=cid:f099af36,vid:V5KrD7CmO4o)
[Rewriting history - from the git pro book](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History)


In order to squash together x many git commits run the following:

```git rebase -i HEAD~3``` to squash together three commits into one.

Then edit the file (its interactive) and change pick to squash. Save the file and quit the text editor. A new text editor session
is automatically started. There you can edit the new combined commit message like you want. Everything that starts with # will 
be ignored.


