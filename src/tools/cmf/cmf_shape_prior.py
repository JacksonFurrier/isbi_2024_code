import matplotlib.pyplot as plt
import point_cloud_utils as pcu
import mcubes
import torch
import numpy as np

from src.algs.arm import lv_indicator
from src.tools.kde.nonlinear_shape_prior import nonlinear_shape_prior_grad

# Some thinking is needed in the Mahalanobis distance part
calc_type = torch.float


def cmf_shape_prior(a_volume, a_opt_params, a_algo_params, a_plot=False, a_save_plot=False):
    """
    Main function for segmenting the left ventricle from a reconstructed 3D volume (scalar-field)

    Args:
        a_volume (N, M, K): array_like
                  To be segmented left ventricle SPECT volume (scalar-field)

        a_opt_params (dict): dict
                      Parameter pack of the optimization. Upper limit of iterations num_iter,
                      err_bound iteration error between steps limit, scaling for the gradient
                      dampening is gamma, steps is the gradient step size

        a_algo_params (dict): dict
                       Parameter pack of the Continuous Max-flow algorithm. TODO

        a_plot (bool): bool
                Bit flag to use plotting of intermediate results or not

        a_save_plot (bool): bool
                     Bit flag to save plot of intermediate results or not
    """
    num_iter, err_bound, gamma, steps = a_opt_params.values()
    u_init, par_lambda, par_nu, c_zero, c_one, b_zero, b_one, z_i, sigma_inv, L, V, sigma_ort, sigma, first_cplx, min_shape_face_count, mean_shape, mean_shape_face = a_algo_params.values()
    m = len(z_i)

    norm_epsilon = 0.001

    # mentioned in the paper
    b_zero = c_zero
    b_one = c_one

    lv_volume = a_volume

    rows, cols, height = a_volume.shape
    im_size = rows * cols * height

    # initialization for CMF
    if a_volume.dtype == torch.int32:
        a_volume = a_volume.astype(calc_type)

    alpha = 2 / (par_lambda + par_nu)

    lv_params = dict(a=1, c=2, sigma=-1)
    f_zero = torch.from_numpy(lv_indicator(a_volume, lv_params))
    f_one = f_zero

    im_eff = (par_lambda / (par_lambda + par_nu)) * a_volume + (par_nu / (par_lambda + par_nu)) \
             * (b_zero * (1 - f_one) + b_one * f_one)

    Cs = (im_eff - c_zero) ** 2
    Ct = (im_eff - c_one) ** 2

    if u_init is None:
        u = torch.where(Cs >= Ct, 1, 0).float()
    else:
        u = torch.where(Cs >= Ct, 1, 0).float() # u_init  # start computation from a precomputed prediction

    ps = torch.minimum(Cs, Ct)
    pt = ps

    pp_y = torch.zeros((rows, cols + 1, height), dtype=calc_type)
    pp_x = torch.zeros((rows + 1, cols, height), dtype=calc_type)
    pp_z = torch.zeros((rows, cols, height + 1), dtype=calc_type)
    div_p = torch.zeros((rows, cols, height), dtype=calc_type)

    cmf_iter = 3
    err_iter = torch.zeros(cmf_iter * num_iter, dtype=calc_type)

    from geomloss import SamplesLoss
    eps = 5 * 1e-2
    loss = SamplesLoss(loss='sinkhorn', p=2, blur=eps)
    sigma = 5 * 1e-1
    k = lambda x, y, sigma: torch.exp(-loss(x, y) / (2 * sigma ** 2))

    if a_plot is True:
        plt.ion()

        figure, axis = plt.subplots(2, 2)
        figure.tight_layout()

        slice_num = torch.int32(a_volume.shape[0] / 2)

        plot_obj_vol = axis[0, 0].imshow(a_volume[slice_num, :, :])
        axis[0, 0].set_title("Left Ventricle Volume")

        plot_obj_seg = axis[0, 1].imshow(f_one[slice_num, :, :])
        axis[0, 1].set_title("Segmentation")

        plot_obj_opt = axis[1, 1].imshow(u[slice_num, :, :])
        axis[1, 1].set_title("Optimality")

        plot_obj_err = axis[1, 0].plot(err_iter[0])
        axis[1, 0].set_title("Iteration error")

        plt.show()

    for i in range(num_iter):
        for j in range(cmf_iter):
            pts = div_p - (ps - pt + u / gamma)

            pp_y[:, 1:-1, :] += steps * (pts[:, 1:, :] - pts[:, :-1, :])
            pp_x[1:-1, :, :] += steps * (pts[1:, :, :] - pts[:-1, :, :])
            pp_z[:, :, 1:-1] += steps * (pts[:, :, 1:] - pts[:, :, :-1])

            # the following steps give the projection to make |p(x)| <= alpha(x)
            squares = pp_y[:, :-1, :] ** 2 + pp_y[:, 1:, :] ** 2
            squares += pp_x[:-1, :, :] ** 2 + pp_x[1:, :, :] ** 2
            squares += pp_z[:, :, :-1] ** 2 + pp_z[:, :, 1:] ** 2

            gk = torch.sqrt(squares * .5)
            gk = (gk <= alpha) + torch.logical_not(gk <= alpha) * (gk / alpha)
            gk = 1 / gk

            pp_y[:, 1:-1, :] = (.5 * (gk[:, 1:, :] + gk[:, :-1, :])) * (pp_y[:, 1:-1, :])
            pp_x[1:-1, :, :] = (.5 * (gk[1:, :, :] + gk[:-1, :, :])) * (pp_x[1:-1, :, :])
            pp_z[:, :, 1:-1] = (.5 * (gk[:, :, 1:] + gk[:, :, :-1])) * (pp_z[:, :, 1:-1])

            div_p = pp_y[:, 1:, :] - pp_y[:, :-1, :]
            div_p += pp_x[1:, :, :] - pp_x[:-1, :, :]
            div_p += pp_z[:, :, 1:] - pp_z[:, :, :-1]

            # update the source flow ps
            pts = div_p + pt - u / gamma + 1 / gamma
            ps = torch.minimum(pts, Cs)

            # update the sink flow pt
            pts = -div_p + ps + u / gamma
            pt = torch.minimum(pts, Ct)

            u_error = gamma * (div_p - ps + pt)
            u -= u_error

            u_error_normed = torch.sum(torch.abs(u_error)) / im_size
            err_iter[cmf_iter * i + j] = u_error_normed

            if a_plot is True:
                plot_obj_opt.set_data(u[slice_num, :, :])
                axis[1, 0].plot(err_iter[0: cmf_iter * i + j])
                plt.draw()

        c_zero = torch.sum((1 - u) * im_eff) / torch.sum(1 - u)
        c_one = torch.sum(u * im_eff) / (torch.sum(u))

        im_mod = c_zero * (1 - u) + c_one * u

        b_zero = torch.sum((1 - f_one) * im_mod) / (torch.linalg.norm(1 - f_one + norm_epsilon) ** 2)
        b_one = torch.sum(f_one * im_mod) / (torch.linalg.norm(f_one + norm_epsilon) ** 2)

        # pos_grad, pos_fac = nonlinear_shape_prior_grad(u.numpy(), sigma_inv, mean_shape, 52, rows)
        zero_volume_boundary(u, a_width=2)
        vert_vol, tri_vol = mcubes.marching_cubes(u.numpy(), 0.5)
        v_decimate, f_decimate, v_correspondence, f_correspondence = pcu.decimate_triangle_mesh(vert_vol,
                                                                                                tri_vol.astype(
                                                                                                    np.int32),
                                                                                                min_shape_face_count)
        z = torch.from_numpy(v_decimate / rows)

        # transform the current shape to the mean shape
        x = mean_shape
        y = z

        N, M, D = x.shape[0], y.shape[0], x.shape[1]  # Number of points, dimension
        p = 2
        blur = 5 * 1e-2

        OT_solver = SamplesLoss(loss="sinkhorn", p=p, blur=blur, reach=1.41, scaling=0.9, debias=False, potentials=True)
        F, G = OT_solver(x, y)  # Dual potentials

        x_i, y_j = x.view(N, 1, D), y.view(1, M, D)
        F_i, G_j = F.view(N, 1), G.view(1, M)

        C_ij = (1 / p) * ((x_i - y_j) ** p).sum(-1)  # (N,M) cost matrix
        eps = blur ** p  # temperature epsilon
        P_ij = ((F_i + G_j - C_ij) / eps).exp() / M  # (N,M) transport plan

        # compute gradient w.r.t. to the aligned "mean" shape|
        proj_mean = torch.reshape(P_ij @ y, mean_shape.shape)
        proj_mean.requires_grad = True  # z could require grad, but it takes forever to compute

        grad_E = nonlinear_shape_prior_grad(V, k, sigma, z_i, proj_mean, L, sigma_ort, first_cplx, m)

        f_one = torch.zeros((rows, cols, height))
        ijk = pcu.voxelize_triangle_mesh(np.reshape(grad_E.detach().numpy(), mean_shape.shape) * rows,
                                         mean_shape_face.astype(np.int32), 1.0, [0., 0., 0.])
        if ijk.ndim == 4:
            f_one[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1
        fill_value = 1
        f_one = torch.from_numpy(flip_vals(pcu.flood_fill_3d(f_one, [0, 0, 0], fill_value), 0, 1))

        print("index: ", i, " energy min: ", grad_E.min(), " energy max: ", grad_E.max(), " shape prior count: ",
              f_one.sum())

        if a_plot is True:
            plot_obj_seg.set_data(f_one[slice_num, :, :])
            plt.draw()

            if a_save_plot is True:
                name = "..\\..\\left_ventricle_" + str(i) + ".png"
                plt.savefig(name, bbox_inches='tight', pad_inches=0)

        im_eff = (par_lambda / (par_lambda + par_nu)) * a_volume + (par_nu / (par_lambda + par_nu)) \
                 * (b_zero * (1 - f_one) + b_one * f_one)
        Cs = (im_eff - c_zero) ** 2
        Ct = (im_eff - c_one) ** 2

    return u, err_iter, num_iter


def flip_vals(A, val1, val2):
    # Find the difference between two values
    diff = val2 - val1

    # Scale masked portion of A based upon the difference value in positive
    # and negative directions and add up with A to have the desired output
    return A + diff * (A == val1) - diff * (A == val2)


def zero_volume_boundary(a_volume, a_width):
    a_volume[:a_width, :, :] = 0
    a_volume[-a_width:, :, :] = 0
    a_volume[:, :a_width, :] = 0
    a_volume[:, -a_width:, :] = 0
    a_volume[:, :, :a_width] = 0
    a_volume[:, :, -a_width:] = 0
