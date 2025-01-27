# CFsOnSphere
## Jain-Kamilla projection in the spherical geometry.

This GitHub repository hosts code implementing the Jain-Kamilla (JK) projection in the spherical geometry, according to the approach introduced in https://arxiv.org/abs/2412.09670, primarily used for the lowest Landau level projection of composite fermion and, more generally parton wavefunctions.

## Installation
1. Users interested in this code will need to first install Julia (https://julialang.org/downloads/).
2. This package is not yet registered with the Julia general registry. Therefore, to use it, the user must first clone this git repository. To use the code provided by this package, the user must then,
   a) if using Julia in the REPL mode, activate the folder as, ```]activate path_to_folder``` (If you navigate to the cloned repo and open the Julia REPL, this would simply be ```]activate .```).
   b) if using Julia in the script mode, i.e. running a Julia script as ```julia myscript.jl```, then you would need to first activate the cloned environment as ```julia --project=path_to_folder myscript.jl```.

## How to use this code? 
