from ngsolve import *
from random import random
import numpy as np
import itertools
import General_Functions as genfunc
import Magnetisation_Functions as magfunc
import time


def strain(u: GridFunction) -> CoefficientFunction:
    """
    Returns the total strain from (Grad(u) + Grad(u)^T) / 2.

    Parameters:
        u (CoefficientFunction): A `CoefficientFunction` (`GridFunction`) or Test function.
    """
    return Sym(Grad(u))


def strain_el(strain_m: CoefficientFunction, u: GridFunction) -> CoefficientFunction:
    """
    Returns the elastic strain, the difference of the total strain and magnetostrain.
    """
    return strain(u) - strain_m


def stress(strain: CoefficientFunction, mu: float, lam: float) -> CoefficientFunction:
    """
    Returns the `stress` associated with (the isotropic) Hooke's law from a given `strain`.

    Parameters:
        strain : The symmetric gradient.
        mu (float): The first lamé coefficient.
        lam (float): The second lamé coefficient.
    """
    return 2 * mu * strain + lam * Trace(strain) * Id(3)


def give_random_displacement(
    disp_grid_func: GridFunction, magnitude: float = 0.01
) -> GridFunction:
    """
    Returns a displacement `GridFunction` with random entries in [-`magnitude`, `magnitude`].

    Parameters:
        disp_grid_func (ngsolve.comp.GridFunction): A `VectorH1` grid function.
        magnitude (float): The largest allowed size for any individual component. Set to 0.01 by default.

    Returns:
        disp_grid_func (ngsolve.comp.GridFunction): A `VectorH1` grid function with randomised nodal values in [-`magnitude`, `magnitude`]^3.
    """
    num_points = genfunc.get_num_nodes(disp_grid_func)
    disp_grid_func.vec.FV().NumPy()[:] = np.random.uniform(
        -magnitude, magnitude, 3 * num_points
    ).flatten()  # 3N uniformly distributed points in [-magnitude, magnitude]
    return disp_grid_func


def give_uniform_displacement(disp_grid_func: GridFunction, direction) -> GridFunction:
    """
    Returns a random normalised magnetisation `GridFunction`.

    Parameters:
        disp_grid_func (ngsolve.comp.GridFunction): A `VectorH1` grid function corresponding to the displacement.\n
        direction (float,float,float): The vector `(x,y,z)` the displacement should have initially.

    Returns:
        disp_grid_func (ngsolve.comp.GridFunction): A uniform `VectorH1` grid function with prescribed direction.
    """
    num_points = genfunc.get_num_nodes(disp_grid_func)
    disp_grid_funcx, disp_grid_funcy, disp_grid_funcz = disp_grid_func.components
    # a, b, c = 2 * random() - 1, 2 * random() - 1, 2 * random() - 1
    a, b, c = direction[0], direction[1], direction[2]
    for i in range(num_points):
        disp_grid_funcx.vec[i] = a
        disp_grid_funcy.vec[i] = b
        disp_grid_funcz.vec[i] = c
    return disp_grid_func


def update_displacement(
    fes_disp: VectorH1,
    disp_gfu: GridFunction,
    disp_gfu_prev: GridFunction,
    strain_m: CoefficientFunction,
    f_body: CoefficientFunction,
    g_surface: CoefficientFunction,
    K: float,
    mu: float,
    lam: float,
) -> GridFunction:
    """

    Updates a displacement vector with the new values.

    Parameters:
        fes_mag (ngsolve.comp.VectorH1): `VectorH1` finite element space.\n
        disp_gfu (ngsolve.comp.GridFunction): The current displacement.\n
        disp_gfu_prev (ngsolve.comp.GridFunction): The displacement from the previous iteration.\n
        f_body (ngsolve.fem.CoefficientFunction): The body force.\n
        g_surface (ngsolve.fem.CoefficientFunction): The traction.

    Returns:
        disp_gfu (ngsolve.comp.GridFunction): The new updated displacement at the next time step.
    """
    new_disp = GridFunction(fes_disp)
    # Test functions
    with TaskManager():
        u = fes_disp.TrialFunction()
        psi = fes_disp.TestFunction()
        # Building the linear system for the displacement
        a_disp = BilinearForm(fes_disp)
        a_disp += InnerProduct(u, psi) * dx  # <u^(i+1), ψ>
        a_disp += (
            K * K * InnerProduct(stress(strain(u), mu, lam), strain(psi)) * dx
        )  # k^2<Cε(u), ε(ψ)>
        f_disp = LinearForm(fes_disp)
        f_disp += (
            K * K * InnerProduct(stress(strain_m, mu, lam), strain(psi)) * dx
        )  # k^2<Cε_m(Π m),ε(ψ)>
        f_disp += (
            InnerProduct(disp_gfu - disp_gfu_prev, psi) * dx
        )  # k<d_t u^i, ψ> = <u^i - u^(i-1), ψ>
        f_disp += InnerProduct(disp_gfu, psi) * dx  # <u^i, ψ>
        f_disp += K * K * InnerProduct(f_body, psi) * dx  # k^2 <f, ψ>
        f_disp += K * K * InnerProduct(g_surface, psi) * ds  # k^2 _/‾ g·ψ ds
        c = Preconditioner(a_disp, "local")
        a_disp.Assemble()
        f_disp.Assemble()
        inv = CGSolver(a_disp.mat, c, maxsteps=1000, precision=1e-8)
        # new_disp.vec.data = (a_disp.mat.Inverse(fes_disp.FreeDofs(), inverse="sparsecholesky")* f_disp.vec)
        new_disp.vec.data = inv * f_disp.vec
    return new_disp


def FIRST_RUN_update_displacement(
    fes_disp: VectorH1,
    disp_gfu: GridFunction,
    vel_gfu: GridFunction,
    strain_m: CoefficientFunction,
    f_body: CoefficientFunction,
    g_surface: CoefficientFunction,
    K: float,
    mu: float,
    lam: float,
) -> GridFunction:
    """
    >**Uses the initial velocity condition instead of a difference quotient.**

    Updates a displacement vector with the new values.

    Parameters:
        fes_mag (ngsolve.comp.VectorH1): `VectorH1` finite element space.\n
        disp_gfu (ngsolve.comp.GridFunction): A `VectorH1` grid function at the i=0 time step.\n
        vel_gfu (ngsolve.comp.GridFunction): A `VectorH1` grid function that models the initial velocity.\n
        strain_m (ngsolve.fem.CoefficientFunction): The magnetostrain.\n
        f_body (ngsolve.fem.CoefficientFunction): The body force.\n
        g_surface (ngsolve.fem.CoefficientFunction): The traction.\n
        K (float): The timestep.\n
        mu (float): The first lamé constant.\n
        lam (float): The second lamé constant.

    Returns:
        new_disp (ngsolve.comp.GridFunction): The new updated displacement at the i=1 time step.
    """
    new_disp = GridFunction(fes_disp)
    with TaskManager():
        u = fes_disp.TrialFunction()
        psi = fes_disp.TestFunction()
        # Building the linear system for the displacement
        a_disp = BilinearForm(fes_disp)
        a_disp += InnerProduct(u, psi) * dx  # <u^(i+1), ψ>
        a_disp += (
            K * K * InnerProduct(stress(strain(u), mu, lam), strain(psi)) * dx
        )  # k^2<Cε(u), ε(ψ)>

        f_disp = LinearForm(fes_disp)
        f_disp += (
            K * K * InnerProduct(stress(strain_m, mu, lam), strain(psi)) * dx
        )  # k^2<Cε_m(Π m),ε(ψ)>
        f_disp += K * InnerProduct(vel_gfu, psi) * dx  # k<d_t u^i, ψ>
        f_disp += InnerProduct(disp_gfu, psi) * dx  # <u^i, ψ>
        f_disp += K * K * InnerProduct(f_body, psi) * dx  # k^2 <f, ψ>
        f_disp += K * K * InnerProduct(g_surface, psi) * ds  # k^2 _/‾ g·ψ ds
        c = Preconditioner(a_disp, "local")
        a_disp.Assemble()
        f_disp.Assemble()
        inv = CGSolver(a_disp.mat, c, maxsteps=1000, precision=1e-8)
        # new_disp.vec.data = (a_disp.mat.Inverse(fes_disp.FreeDofs(), inverse="sparsecholesky")* f_disp.vec)
        new_disp.vec.data = inv * f_disp.vec
    return new_disp


def elastic_energy(
    mesh: Mesh,
    disp_gfu: GridFunction,
    strain_m: CoefficientFunction,
    f_body: CoefficientFunction,
    g_surface: CoefficientFunction,
    KAPPA: float,
    mu: float,
    lam: float,
) -> float:
    """
    Computes the elastic energy\n
    > κ/2 ∫(ε_el)C:(ε_el) dx - κ∫ f.u dx - κ∫g.u ds

    Parameters:
        mesh (ngsolve.comp.Mesh): Displacement FE mesh.\n
        disp_gfu (ngsolve.comp.GridFunction): Displacement `VectorH1` grid function.\n
        strain_m (ngsolve.fem.CoefficientFunction): The magnetostrain.\n
        f_body (ngsolve.fem.CoefficientFunction): The body force.\n
        g_surface (ngsolve.fem.CoefficientFunction): The traction.\n
        KAPPA (float): Relative strength of magnetic to elastic contributions.\n
        mu (float): The first lamé constant.\n
        lam (float): The second lamé constant.

    Returns:
        energy (float): Elastic energy.
    """
    mystrain = strain_el(strain_m, disp_gfu)
    vol_integrand = 0.5 * InnerProduct(
        stress(mystrain, mu, lam), mystrain
    ) - InnerProduct(
        f_body, disp_gfu
    )  # 1/2 <Cε_el(m,u), ε_el(m,u)> - <f,u>
    surf_integrand = InnerProduct(g_surface, disp_gfu)  # -<g,u>_BND
    energy = Integrate(vol_integrand, mesh, VOL)
    energy += -Integrate(surf_integrand, mesh, BND)
    return KAPPA * energy


def initial_kinetic_energy(mesh: Mesh, vel_gfu: GridFunction, KAPPA: float) -> float:
    """
    Returns the kinetic energy κ/2 ∫ ||u_t||^2 dx.

    Parameters:
        mesh (ngsolve.comp.Mesh): Displacement FE mesh.\n
        vel_gfu (ngsolve.comp.GridFunction): Initial velocity VectorH1 grid function.\n
        KAPPA (float): Relative strength of magnetic to elastic contributions.\n

    Returns:
        Kinetic energy (float): The initial kinetic energy.
    """
    integrand = InnerProduct(vel_gfu, vel_gfu)
    return KAPPA * 0.5 * Integrate(integrand, mesh, VOL)


def kinetic_energy(
    mesh: Mesh,
    disp_gfu: GridFunction,
    disp_gfu_prev: GridFunction,
    KAPPA: float,
    K: float,
) -> float:
    """
    Returns the kinetic energy κ/(2K^2) ∫ ||u^i - u^(i-1)||^2 dx.

    Parameters:
        mesh (ngsolve.comp.Mesh): Displacement FE mesh.\n
        disp_gfu (ngsolve.comp.GridFunction): The ith displacement.\n
        disp_gfu_prev (ngsolve.comp.GridFunction): The (i-1)th displacement.\n
        KAPPA (float): Relative strength of magnetic to elastic contributions.\n
        K (float): Timestep.

    Returns:
        Kinetic energy (float): The kinetic energy.
    """
    disp_diff = disp_gfu - disp_gfu_prev
    integrand = InnerProduct(disp_diff, disp_diff)
    return KAPPA * 0.5 * Integrate(integrand, mesh, VOL) / (K * K)


def Voigt_6x6_to_full_3x3x3x3(C: np.ndarray) -> np.ndarray:
    """
    Convert from the Voigt representation of the stiffness matrix to the full
    3x3x3x3 representation.
    Parameters
    ----------
    C : array_like
        6x6 stiffness matrix (Voigt notation).
    Returns
    -------
    C : array_like
        3x3x3x3 stiffness matrix.
    Taken from https://github.com/libAtoms/matscipy/blob/master/matscipy/elasticity.py
    """

    C = np.asarray(C)
    C_out = np.zeros((3, 3, 3, 3), dtype=float)
    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        Voigt_i = full_3x3_to_Voigt_6_index(i, j)
        Voigt_j = full_3x3_to_Voigt_6_index(k, l)
        C_out[i, j, k, l] = C[Voigt_i, Voigt_j]
    return C_out


def full_3x3_to_Voigt_6_index(i: int, j: int) -> int:
    """
    Taken from https://github.com/libAtoms/matscipy/blob/master/matscipy/elasticity.py
    """
    if i == j:
        return i
    return 6 - i - j


def isotropic_isochoric_voigt_array(lambda_m: float) -> np.ndarray:
    """
    Returns the Voigt (6x6) matrix representation of the isotropic and isochoric magnetostriction tensor.
    Parameters:
        lambda_m (float): The saturation magnetostrictive strain.

    Returns:
        myarray (np.ndarray): The Voigt (6x6) matrix representation.
    """
    myarray = np.zeros((6, 6))
    myarray[0, 0] = lambda_m
    myarray[1, 1] = lambda_m
    myarray[2, 2] = lambda_m
    myarray[3, 3] = 0.75 * lambda_m
    myarray[4, 4] = 0.75 * lambda_m
    myarray[5, 5] = 0.75 * lambda_m
    myarray[1, 2] = -0.5 * lambda_m
    myarray[1, 3] = -0.5 * lambda_m
    myarray[2, 1] = -0.5 * lambda_m
    myarray[3, 1] = -0.5 * lambda_m
    return myarray


def Z_tensor(lambda_m: float) -> np.ndarray:
    return Voigt_6x6_to_full_3x3x3x3(isotropic_isochoric_voigt_array(lambda_m))


def magnetostriction_field(
    strainu: CoefficientFunction,
    proj_mag_gfu: GridFunction,
    mu: float,
    lam: float,
    lambda_m: float,
) -> CoefficientFunction:
    """
    Takes in a strain and magnetisation, and returns (assuming Z is symmetric, isotropic and isochoric)
    2 ZC[ε(u) - ε_m(Proj(m))] Proj(m)
    """
    strain_m = magfunc.build_strain_m(proj_mag_gfu, lambda_m)  # ε_m(Proj(m))
    myStress = stress(strainu - strain_m, mu, lam)  # C[ε(u) - ε_m(Proj(m))]
    magStress = (
        3
        * lambda_m
        / 2
        * (
            myStress - Trace(myStress) * Id(3) / 3
        )  # was previously missing a factor of 1/3 in the Id(3) term
    )  #  ZC[ε(u) - ε_m(Proj(m))]
    return 2 * magStress * proj_mag_gfu  # 2 ZC[ε(u) - ε_m(Proj(m))] Proj(m)
