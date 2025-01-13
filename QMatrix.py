from ngsolve import *
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import time
import Magnetisation_Functions as magfunc
import General_Functions as genfunc


def qmatrix(mag_gfu: GridFunction, fes_mag: VectorH1) -> scipy.sparse.csr.csr_matrix:
    """
    Parameters:
        mag_grid_func (ngsolve.comp.GridFunction): The magnetisation degrees of freedom.
        fes_mag (ngsolve.comp.VectorH1): `VectorH1` finite element space.

    Returns:
        Q (scipy.sparse.csr.csr_matrix): A 2N by 3N matrix for use in the null space method solver.

    Even rows should contain up to two non-zero elements, and odd rows should contain up to three non-zero elements
    """
    mag_grid_func = magfunc.nodal_projection(mag_gfu, fes_mag)
    m1, m2, m3 = mag_grid_func.components
    m1, m2, m3 = m1.vec.FV().NumPy(), m2.vec.FV().NumPy(), m3.vec.FV().NumPy()
    column_m = np.column_stack(
        (m1, m2, m3)
    )  # contains the other pair of basis vectors for every node

    out_array = q_basis(column_m)
    block_me_please = []
    for i in [0, 1, 2]:  # build a 2*N by N block for each column of column_m
        block_me_please.append(q_block(out_array[:, i]))
    my_qmatrix = scipy.sparse.bmat([block_me_please], format="csr")
    return my_qmatrix


def q_basis(m: np.ndarray) -> np.ndarray:
    """
    Given a magnetisation vector m = [[m1(z_1), m2(z_1), m3(z_1)],
                                     [m1(z_2), m2(z_2), m3(z_2)],
                                     ...,
                                     [m1(z_N), m2(z_N), m3(z_N)]],
    returns a suitable basis based upon the smallest in magnitude of m1, m2, m3 in the form
    [[L1(z1),L2(z1),L3(z1)],
     [M1(z1),M2(z1),M3(z1)],
     [L1(z2),L2(z2),L3(z2)],
     [M1(z2),M2(z2),M3(z2)],
     ...]
    """
    N = len(m)
    index = np.argmin(np.abs(m), axis=1)  # yields which case, 0,1,2 should be used
    out_array = np.zeros((2 * N, 3))
    for i in range(0, N):
        u, v, w = m[i]
        L, M = basis_choice(u, v, w, index[i])
        out_array[2 * i] = L
        out_array[2 * i + 1] = M

    return out_array


def basis_choice(u: float, v: float, w: float, index: int):
    """
    We use notation from Ramage 2013 during this section.\n
    Given an input magnetisation vector n=[u,v,w]^T, return two vectors L,M such that (n,L,M) is an orthonormal basis.\n
    This is slow due to the use of if/else statements, which check each row.\n
    It may be possible to vectorize.
    """
    if index == 0:
        square = v * v + w * w
        my_norm = np.sqrt(square)
        L = np.array([0, -w, v]) / my_norm
        M = np.array([square, -u * v, -u * w]) / my_norm
        return L, M
    elif index == 1:
        square = u * u + w * w
        my_norm = np.sqrt(square)
        L = np.array([w, 0, -u]) / my_norm
        M = np.array([-v * u, square, -v * w]) / my_norm
        return L, M
    elif index == 2:
        square = u * u + v * v
        my_norm = np.sqrt(square)
        L = np.array([-v, u, 0]) / my_norm
        M = np.array([-w * u, -w * v, square]) / my_norm
        return L, M


def q_block(my_vec: np.ndarray) -> np.ndarray:
    """
    Given a 2Nx1 column of the form   = [[L1],
                                        [M1],
                                        ...,
                                        [LN],
                                        [MN]],
    returns a sparse block of size 2NxN with the ith column having the 2i-1 and 2i positions
    filled with the i and i+1 elements of the vector. That is,
    [[L1,0,0,0,...]
     [M1,0,0,0,...]
     [0,L2,0,0,...]
     [0,M2,0,0,...]
     ...
     ...0,0,0,0,LN],
     ...0,0,0,0,MN]],
    """
    assert len(my_vec) % 2 == 0
    N = len(my_vec) // 2
    cols = np.repeat(np.arange(N), 2)  # looks like [0,0,1,1,...,N-1,N-1]
    rows = np.arange(2 * N)  # looks like [0,1,2,3,...,2N-1]
    the_q_block = scipy.sparse.csr_matrix((my_vec, (rows, cols)))
    return the_q_block


def give_q_magnetisation_update(
    A: BilinearForm,
    B: np.ndarray,
    F: LinearForm,
    M_FIXED: BilinearForm,
    L_FIXED: BilinearForm,
    Q: scipy.sparse.csr.csr_matrix,
    v0: np.ndarray,
) -> np.ndarray:
    """
    Returns the tangent plane update v^(i) to the magnetisation such that m^(i+1) = m^(i) + v^(i).

    Parameters:
        A (ngsolve.comp.BilinearForm): The 3Nx3N assembled magnetisation "stiffness" matrix from the variational formulation.\n
        B (ngsolve.bla.MatrixD): The Nx3N tangent plane matrix.\n
        F (ngsolve.comp.LinearForm): The 3Nx1 assembled force vector from the variational formulation.\n
        M_FIXED (ngsolve.comp.BilinearForm): Fixed mass matrix\n
        L_FIXED (ngsolve.comp.BilinearForm): Fixed skew matrix\n
        Q (np.ndarray): The null space matrix. Reduces problem from 3Nx3N to 2Nx2N\n
        v0 (np.ndarray): Initial trial tangent vector. Used in GMRES.

    Returns:
        v (numpy.ndarray): The set of components to use for the update.
    """
    # convert to sparse matrices and add together
    rows, cols, vals = A.mat.COO()
    A = scipy.sparse.csr_matrix((vals, (rows, cols)))
    rows, cols, vals = M_FIXED.mat.COO()
    M_FIXED = scipy.sparse.csr_matrix((vals, (rows, cols)))
    rows, cols, vals = L_FIXED.mat.COO()
    L_FIXED = scipy.sparse.csr_matrix((vals, (rows, cols)))
    A += M_FIXED + L_FIXED
    F = F.vec.FV().NumPy()[:]
    N = len(F) // 3
    # rotate to Q basis
    stiffness_block = Q * A * Q.T
    force_block = Q * F
    time2 = time.time()
    # M = scipy.sparse.linalg.LinearOperator((2 * N, 2 * N), my_precon.solve)
    M2 = scipy.sparse.linalg.spilu(stiffness_block)
    M = scipy.sparse.linalg.LinearOperator((2 * N, 2 * N), M2.solve)
    z, myinfo = scipy.sparse.linalg.gmres(
        stiffness_block, force_block, rtol=1e-8, x0=Q * v0, M=M
    )  # solve (QAQ^T)z = QF using GMRES with initial condition from previous
    time3 = time.time()
    # rotate z out of Q basis
    v = Q.T * z
    residual = np.linalg.norm(
        B.dot(v), np.inf
    )  # in theory, the update should satisfy |Bv| = 0.
    print(f"Q GMRES completed in {time3-time2}, info={myinfo}, residual = {residual}")
    if residual > 1e-8:
        print(
            f"WARNING: |Bv| = {residual} > 1e-8. Tangent plane matrix B or update v may not be correctly calculated."
        )
    return v
