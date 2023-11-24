import torch
import torchvision.transforms as torch_transform

from pykeops.torch import Vi, Vj
from pykeops.torch import LazyTensor

from skimage import measure
import point_cloud_utils as pcu
import mcubes

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def nonlinear_shape_prior(shape_priors, sigma, num_verts):
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
    # compute kernel embedding
    from geomloss import SamplesLoss
    eps = 5 * 1e-2
    loss = SamplesLoss(loss='sinkhorn', p=2, blur=eps)

    m = shape_priors.shape[0]
    # k = lambda x, y, sigma : GaussKernel(x, y, sigma)
    k = lambda x, y, sigma: torch.exp(-loss(x, y) / (2 * sigma ** 2))
    sigma = 5 * 1e-3

    E = (1 / m) * torch.ones([m, m])
    K = torch.zeros([m, m])

    height, width, depth = shape_priors[0].shape
    z_i = []
    shape_face_count = torch.zeros([m], dtype=torch.int32)
    shape_faces = []
    for i in range(m):
        verts_shape, tri_shape = mcubes.marching_cubes(shape_priors[i], 0.5)
        z_i.append(torch.from_numpy(verts_shape) / depth)
        shape_faces.append(tri_shape)
        shape_face_count[i] = tri_shape.shape[0]

    min_shape_face_count = shape_face_count.min()
    # if k_til is wrongfully implemented or slow, or numerically unstable,
    # then one can use K_til̃ = K − KE − EK + EKE
    mean_shape = z_i[int(m / 2)]  # try it with Wasserstein barycenter here compute the mean shape
    mean_shape_face = shape_faces[int(m / 2)]  # save the faces as well

    for i in range(m):
        for j in range(m):
            K[i, j] = k(z_i[i], z_i[j], sigma)

    K_til = K - K @ E - E @ K + E @ K @ E

    # keep only real eigenvalues and eigenvectors
    L, V = torch.linalg.eigh(K_til)
    L = torch.flip(L, [0])
    V = torch.flip(V, [0, 1])
    print(L)

    first_cplx = torch.where(L <= 1)[0][0]
    sigma_ort = L[first_cplx - 1] / 2

    L[first_cplx:] = 0.0
    V[:, first_cplx:] = 0.0
    reg_mx = torch.eye(K.shape[0])

    Sigma_ort = V @ torch.diag(L) @ V.t() + sigma_ort * (reg_mx - V @ V.t())

    return z_i, torch.linalg.inv(
        Sigma_ort), L, V, sigma_ort, sigma, first_cplx, min_shape_face_count, mean_shape, mean_shape_face


# implement something similar to [1] https://projects.ics.forth.gr/cvrl/publications/conferences/2000_eccv_SVD_jacobian.pdf
# some try-outs for understanding for 2x2 linear systems
def nonlinear_shape_prior_grad(V, kernel, sigma, z_i, z, L, L_ort, r, m):
    loss = 0.0
    # optimized gradient computation
    par_z = torch.zeros([m, *z.shape])
    for i in range(m):
        par_z[i] += torch.autograd.grad(kernel(z_i[i], z, sigma), [z])[0]
        for k in range(m):
            par_z[i] -= (1 / m) * torch.autograd.grad(kernel(z, z_i[k], sigma), [z])[0]

    for k in range(r):
        for i in range(m):
            loss += (V[k, i] * par_z[i]) ** 2
        loss *= (L[k] ** (-1) - L_ort ** (-1))

    par_zz = torch.autograd.grad(kernel(z, z, sigma), [z])[0]
    for k in range(m):
        par_zz -= 2.0 * (1 / m) * torch.autograd.grad(kernel(z, z_i[k], sigma), [z])[0]

    loss += (L_ort ** (-1)) * par_zz
    return 2.0 * loss


def GaussKernel(sigma):
    x, y, b = Vi(0, 2), Vj(1, 2), Vj(2, 2)
    gamma = 1 / (2 * sigma * sigma)
    D2 = x.sqdist(y) / (2 * 64 * 64)
    K = (-D2 * gamma).exp()
    return ((0.3989 / sigma) * K * b).sum_reduction(axis=1)
