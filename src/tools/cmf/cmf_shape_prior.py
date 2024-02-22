import matplotlib.pyplot as plt
import point_cloud_utils as pcu
import mcubes
import torch
import numpy as np
from skimage import measure
from scipy.spatial.distance import cdist
from simpleicp import PointCloud, SimpleICP

from src.algs.arm import lv_indicator
from src.tools.kde.nonlinear_shape_prior import E_phi_grad_opt, E_phi_grad

# Some thinking is needed in the Mahalanobis distance part
calc_type = torch.float
centering_point = np.array([0.45, 0.45, 0.45])


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
    par_lambda, par_nu, c_zero, c_one, b_zero, b_one, z_i, sigma_inv, L, V, sigma_ort, sigma, first_cplx, min_shape_face_count, mean_shape, mean_shape_face, k_matrix_sum, k_matrix, kernel = a_algo_params.values()
    m = len(z_i)

    norm_epsilon = 0.001

    # mentioned in the paper
    b_zero = c_zero
    b_one = c_one

    volume = a_volume
    density_vol = volume / volume.sum()

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

    u_prev = torch.zeros([rows, cols, height])
    u = torch.where(Cs >= Ct, 1, 0).float()

    ps = torch.minimum(Cs, Ct)
    pt = ps

    pp_y = torch.zeros((rows, cols + 1, height), dtype=calc_type)
    pp_x = torch.zeros((rows + 1, cols, height), dtype=calc_type)
    pp_z = torch.zeros((rows, cols, height + 1), dtype=calc_type)
    div_p = torch.zeros((rows, cols, height), dtype=calc_type)

    cmf_iter = 3
    err_iter = torch.zeros(cmf_iter * num_iter, dtype=calc_type)
    norm_u_iter = torch.zeros(num_iter + 1, dtype=calc_type)

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

        norm_u_iter[i + 1] = torch.linalg.norm(u)

        c_zero = torch.sum((1 - u) * im_eff) / torch.sum(1 - u)
        c_one = torch.sum(u * im_eff) / (torch.sum(u))

        im_mod = c_zero * (1 - u) + c_one * u

        b_zero = torch.sum((1 - f_one) * im_mod) / (torch.linalg.norm(1 - f_one + norm_epsilon) ** 2)
        b_one = torch.sum(f_one * im_mod) / (torch.linalg.norm(f_one + norm_epsilon) ** 2)

        print("u sum: ", u.sum(), "u max: ", u.max(), "u min: ", u.min(), "u count:", (u > 0).sum())

        zero_volume_boundary(u, a_width=2)

        vert_vol, tri_vol, _, _ = measure.marching_cubes(u.numpy(), 0.1)
        cv, nv, cf, nf = pcu.connected_components(vert_vol, tri_vol.astype(np.int32))

        num_components = nv.size
        print("Connected components: ", num_components)
        print("Iteration: ", i, "Norm diff: ", torch.abs(norm_u_iter[i + 1] - norm_u_iter[i]))

        f_one = torch.zeros((rows, cols, height))

        component = 0

        while (component < num_components) and (torch.abs(norm_u_iter[i + 1] - norm_u_iter[i]) < 0.178) or (
                i + 1) == num_iter:  # 3 for parallel images
            nu = 1e-2

            if component >= num_components:
                break

            if num_components > 1:
                component_face_count = nf[component]
            else:
                component_face_count = nf

            v_decimate, f_decimate, v_correspondence, f_correspondence = \
                pcu.decimate_triangle_mesh(vert_vol, tri_vol[cf == component].astype(np.int32),
                                           min(min_shape_face_count.numpy(), component_face_count))

            z = torch.from_numpy(v_decimate / cols)
            z.requires_grad = True

            # renorm to size 1 and translate it to center_point
            z_dist = cdist(z.detach().numpy(), z.detach().numpy(), 'euclidean')
            max_real_size = z_dist.max() * cols
            if max_real_size <= 20:  # dummy "size" selection 15 for parallel geometries, 20 for mph -> sharpen this
                print("Skipping object with diameter: ", max_real_size)
                component = component + 1
                continue

            z_scaled = z * (1.0 / (z_dist.max()))
            z_translation = z_scaled.mean(dim=0) - torch.from_numpy(centering_point)

            # Project current shape on the mean shape as in [1]
            pc_fix = PointCloud(mean_shape.detach().numpy(), columns=["x", "y", "z"])
            pc_mov = PointCloud((z_scaled - z_translation).detach().numpy(), columns=["x", "y", "z"])
            icp = SimpleICP()
            icp.add_point_clouds(pc_fix, pc_mov)
            H, proj_mean_icp, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=1)
            # add reorientation based registration here

            proj_mean = torch.from_numpy(proj_mean_icp)
            proj_mean.requires_grad = True

            grad_E = E_phi_grad_opt(V, kernel, k_matrix, k_matrix_sum, sigma, z_i, proj_mean, L, sigma_ort, first_cplx,
                                    m)

            # the last terms in the gradient calculation
            # d til_z / d_z_c * d_z_c / d_z
            Rot = H[:-1, :-1]
            translation = H[:-1, -1]

            it_shape = ((proj_mean - nu * grad_E - torch.from_numpy(translation)) @ torch.from_numpy(
                Rot) + z_translation) * z_dist.max()
            print("Translation:", translation, "Normalization factor: ", z_dist.max(), "Mean translation: ",
                  z_translation)

            # voxelization and bounds checking
            ijk = pcu.voxelize_triangle_mesh((it_shape).detach().numpy() * cols, f_decimate.astype(np.int32), 1,
                                             [0., 0., 0.])
            ijk = ijk[np.sum(np.logical_and(ijk >= 0, ijk < cols), axis=1) == 3, :]
            ijk = ijk[ijk[:, 0] < rows]  # more likely that the axial dim is different

            f_one[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1

            print("component: ", component, " energy min: ", grad_E.min(), " energy max: ", grad_E.max(),
                  " shape prior count: ", f_one.sum(), " shape prior mean pos:", (it_shape).mean())

            component = component + 1

        if a_plot is True:
            plot_obj_seg.set_data(f_one[slice_num, :, :])
            plt.draw()

            if a_save_plot is True:
                name = "..\\..\\left_ventricle_" + str(i) + ".png"
                plt.savefig(name, bbox_inches='tight', pad_inches=0)

        im_eff = (par_lambda / (par_lambda + par_nu)) * a_volume + (par_nu / (par_lambda + par_nu)) \
                 * (b_zero * (1 - f_one) + b_one * f_one)

        H = torch.where(u > 0, 1, 0)
        Cs = (im_eff - c_zero) ** 2 * torch.kl_div(H * density_vol, (1 - H) * density_vol).sum()
        Ct = (im_eff - c_one) ** 2 * torch.kl_div((1 - H) * density_vol, H * density_vol).sum()

    return u, err_iter, num_iter, f_one


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
