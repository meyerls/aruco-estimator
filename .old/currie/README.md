This doesnt use the module code and wont be maintained as part of it. 

By directly calling a colmap process and performing reconstruction,
* it breaks the contract of the module, which is to perform registration and scaling tasks on precomputed reconstruction projects.
* It adds a major dependency (colmap, pycolmap) without general benefit.

I've left it accesible as I didnt properly review and call it out during your PR. 
