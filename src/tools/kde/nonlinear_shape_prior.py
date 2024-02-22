import torch
import torchvision.transforms as torch_transform

from pykeops.torch import Vi, Vj
from pykeops.torch import LazyTensor

from skimage import measure
from scipy.spatial.distance import cdist
import point_cloud_utils as pcu
import mcubes
import copy as cp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def nonlinear_shape_prior(shape_priors, kernel, sigma, centering_point):
    """
    Nonlinear statistics shape prior based on kernel density estimation in the feature space
        [1] Shape statistics in kernel space for variational image segmentation - Daniel Cremers, Timo Kohlberger,
                                                                                  Christoph Schnoerr
        [2] Active Shape Models - Their Training and Application - T. F. Cootes, C. J. Taylor, D. H. Cooper, J. Graham

    Args:
        z:
        z_i:
        sigma:

    Returns:
        energy:
    """
    m = shape_priors.shape[0]

    E = (1 / m) * torch.ones([m, m], dtype=torch.float64)
    K = torch.zeros([m, m], dtype=torch.float64)

    height, width, depth = shape_priors[0].shape
    z_i = []
    shape_face_count = torch.zeros([m], dtype=torch.int32)
    shape_faces = []
    for i in range(m):
        verts_shape, tri_shape = mcubes.marching_cubes(shape_priors[i], 0.0)
        cur_prior_shape = verts_shape / depth

        # set mesh size to 1 and move it to the centering point
        verts_dist = cdist(cur_prior_shape, cur_prior_shape, 'euclidean')
        verts_scaled = cur_prior_shape * 1.0 / verts_dist.max()
        verts_scaled_translation = centering_point - verts_scaled.mean(axis=0)
        verts_translated = verts_scaled + verts_scaled_translation

        z_i.append(torch.from_numpy(verts_translated))
        shape_faces.append(tri_shape)
        shape_face_count[i] = tri_shape.shape[0]

    min_shape_face_count = shape_face_count.min()
    # if k_til is wrongfully implemented or slow, or numerically unstable,
    # then one can use K_til̃ = K − KE − EK + EKE
    mean_shape = z_i[int(m / 2)]  # try it with Wasserstein barycenter here compute the mean shape
    mean_shape_face = shape_faces[int(m / 2)]  # save the faces as well

    for i in range(m):
        for j in range(m):
            K[i, j] = kernel(z_i[i], z_i[j], sigma)

    K_til = K - K @ E - E @ K + E @ K @ E

    # keep only real eigenvalues and eigenvectors
    L, V = torch.linalg.eigh(K_til)
    L = torch.flip(L, [0])
    V = torch.fliplr(V)

    limit_val = 1e-6
    if (L <= limit_val).any():
        first_cplx = torch.where(L <= limit_val)[0][0]
        sigma_ort = L[first_cplx - 1] / 2.0

        L[first_cplx:] = 0.0
        V[:, first_cplx:] = 0.0
        reg_mx = torch.eye(K.shape[0])

        Sigma_ort = V @ torch.diag(L) @ V.t() + sigma_ort * (reg_mx - V @ V.t())
    else:  # bad bad things happen
        first_cplx = -1
        sigma_ort = 1
        Sigma_ort = V @ torch.diag(L) @ V.t()

    return z_i, torch.linalg.inv(
        Sigma_ort), L, V, sigma_ort, sigma, first_cplx, min_shape_face_count, mean_shape, mean_shape_face, K.sum(), K


# implement something similar to [1] https://projects.ics.forth.gr/cvrl/publications/conferences/2000_eccv_SVD_jacobian.pdf
# some try-outs for understanding for 2x2 linear systems
def E_phi_grad(V, kernel, k_matrix_sum, sigma, z_i, z, L, L_ort, r, m):
    loss = torch.zeros(z.shape)
    # optimized gradient computation
    par_z = torch.zeros([m, *z.shape])
    k_til = torch.zeros([m])
    for i in range(m):
        par_z[i] += torch.autograd.grad(kernel(z_i[i], z, sigma), [z])[0]
        k_til[i] += kernel(z_i[i], z, sigma)
        for k in range(m):
            par_z[i] -= (1 / m) * torch.autograd.grad(kernel(z, z_i[k], sigma), [z])[0]
            k_til[i] -= (1 / m) * (kernel(z, z_i[k], sigma) + kernel(z_i[i], z_i[k], sigma))

    k_til += (1 / (m ** 2)) * k_matrix_sum

    alpha = cp.copy(V)
    alpha[:, :r] *= (torch.sqrt(L[:r])[:, None]).t()

    for k in range(r):
        for i in range(m):
            loss += (alpha[i, k] * k_til[i]) * (alpha[i, k] * par_z[i]) * (L[k] ** (-1) - L_ort ** (-1))

    par_zz = torch.zeros([*z.shape])
    for k in range(m):
        par_zz -= (1 / m) * torch.autograd.grad(kernel(z, z_i[k], sigma), [z])[
            0]  # multiplication with 2 is missing
    loss += (L_ort ** (-1)) * par_zz

    return 2.0 * loss

def E_phi_grad_opt(V, kernel, k_m, k_matrix_sum, sigma, z_i, z, L, L_ort, r, m):
    loss = torch.zeros(z.shape)

    # lightspeed optimized gradient computation
    par_z = torch.zeros([m, *z.shape])
    kernel_ = torch.zeros([m])
    for i in range(m):
        par_z[i] = torch.autograd.grad(kernel(z_i[i], z, sigma), [z])[0]
        kernel_[i] = kernel(z_i[i], z, sigma)

    k_til = kernel_ - (1 / m) * kernel_.sum(dim=0) - (1 / m) * k_m.sum(dim=1) + (1 / (m ** 2)) * k_matrix_sum

    par_z_sum = (1 / m) * par_z.sum(dim=0)
    kernel_til = lambda par_z, index: par_z[index] - par_z_sum

    alpha = cp.copy(V)
    alpha[:, :r] *= (torch.sqrt(L[:r])[:, None]).t()

    for k in range(r):
        for i in range(m):
            loss += (alpha[i, k] * k_til[i]) * (alpha[i, k] * kernel_til(par_z, i)) * (L[k] ** (-1) - L_ort ** (-1))

    par_zz = torch.zeros([*z.shape])
    for k in range(m):
        par_zz -= (1 / m) * par_z[k]
    loss += (L_ort ** (-1)) * par_zz

    return 2.0 * loss


def GaussKernel(sigma):
    x, y, b = Vi(0, 2), Vj(1, 2), Vj(2, 2)
    gamma = 1 / (2 * sigma * sigma)
    D2 = x.sqdist(y) / (2 * 64 * 64)
    K = (-D2 * gamma).exp()
    return ((0.3989 / sigma) * K * b).sum_reduction(axis=1)
