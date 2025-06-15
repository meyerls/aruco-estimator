This doesnt use the module code and wont be maintained as part of it. By directly calling a colmap process and performing reconstruction, it breaks the contract of the module, which is to perform registration tasks on precomputed reconstruction projects. I've left it accesible as I didnt call it out during your PR.

There are colmap dependencies throughout the project which I'll be isolating before the next PyPi release.
