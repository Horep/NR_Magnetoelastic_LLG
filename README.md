# DecoupledLLGMagnetostrictionIntegrator
This uses NGSolve/Netgen to solve the Landau-Lifshitz-Gilbert equation including magnetostriction, coupled with the conservation of momentum equation.

We use a tangent plane scheme without a projection step.

# Important information:

Written for Python 3.12. The previous versions may work, but no guarantees, and you are likely to get different behaviour in some areas. In particular, there was a change in scipy's GMRES solver, from "tol" to "rtol" and "atol".

Documentation of various functions is included within the auxiliary files, `Elastic_Functions.py`, `General_Functions.py` and `Magnetisation_Functions.py`. I do not recommend touching `QMatrix.py` for any reason.

EXPERIMENTS:

To run experiments from the associated paper, there are two mesh builders included.
One is for building cuboid meshes, and the other is for the hemisphere.
The units are assuming exchange lengths of the reference material,
so if the exchange length is 1nm, then for a 6nm x 6nm x 6nm material, input (6,6,6) for the corner furthest from zero.

The default experiment is set up to run the nutation experiment.

You will need to input the `H_MAX` parameter for different meshes. This can be difficult to match up as ngsolve's internal system does not assume this as a true "maximum length".
Generally, pick a "nice number" nearby.

e.g. to get h_max ~ 0.209, inputting `H_MAX = 0.25` should suffice.

If you would like it as a Python file instead of a Jupyter file, this can be done in VSCODE with CTRL+SHIFT+P, and converting to a Python file.

For Dirichlet conditions, it is currently set up on the `left` boundary box. The cuboid has a name on every face,
so you can pick and choose if you would like.

For the Zeeman field, you change the `zeeman_factor` parameter. If there is no field, set the inside of the `Parameter` to `0.0`. If you replace this with a zero without the parameter wrapping, ngsolve will fail to understand it.