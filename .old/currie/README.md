This doesnt use the module code and wont be maintained as part of it. By directly calling a colmap process and performing reconstruction, it breaks the contract of the module, which is to perform registration tasks on arbitrary reconstruction projects. I've left it accesible as I didnt call it out during your PR.

I have placed some of my own code in here, which I developed for my project on my fork, but breaks the contract of the module.

There are colmap dependencies throughout the project which I'll be isolating before the next PyPi release.