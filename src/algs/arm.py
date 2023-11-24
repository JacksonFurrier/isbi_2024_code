import numpy as np
import numpy.linalg as la

from scipy import ndimage
from scipy import signal

from skimage.morphology import cube

import matplotlib.pyplot as plt

filter_size = 2
filter_kernel = cube(filter_size) / np.size(cube(filter_size))
calc_type = np.float32


def rotation_mx(a_deg):
    """
    Create rotation matrix in 3D for the model to transform the prior information.

    Args:
        a_deg (scalar): scalar
               Angle describing the rotation for the cardiac model

    Returns:
        rot_mx (3, 3): array_like
                Rotation matrix for the 3D transformation
    """
    alpha, beta, gamma = a_deg  # it is in radian already

    rot_mx = np.zeros([3, 3])

    rot_mx[0, 0] = np.cos(alpha) * np.cos(beta)
    rot_mx[1, 0] = np.sin(alpha) * np.cos(beta)
    rot_mx[2, 0] = -np.sin(beta)

    rot_mx[0, 1] = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma)
    rot_mx[1, 1] = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma)
    rot_mx[2, 1] = np.cos(beta) * np.sin(gamma)

    rot_mx[0, 2] = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)
    rot_mx[1, 2] = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)
    rot_mx[2, 2] = np.cos(beta) * np.cos(gamma)

    return rot_mx


def lv_indicator(a_volume, a_params, a_transformation_params=[np.eye(3, 3), [0, 0, 0], 0], a_plot=False, a_plot_profile=False):
    """
    Generating left ventricle indicator function R^3 -> R. 

    Args:
        a_volume (N, M, K): array_like
                  Generally it just enpacking the sizes that are needed, it is kept if more information
                  might needed, e.g.: activity under the indicator function
        a_params (dict): dict
                  Parameters exposing the model, where a, c is the epicardial and a is the endocardial
                  ``surface''. Slicing limit of the ellipsoid, which basically describes the slicing plane,
                  where it intersects the ellipsoid-ish model
        a_transformation_params (3,3): array_like
                                 Transformation params of the indicator function to help the caller in transforming
                                 the model.

    Returns:
        lv_indicator_function (N, M, K): array_like
                               The indicator function of the model based on the parameters.
                  
    """
    a, c, sigma = a_params.values()
    rot_mx, trans, scale = a_transformation_params

    a_center = np.floor(0.5 * np.asarray(a_volume.shape))  # setting the heart model in the middle of the image\
    # (TODO: put it at a random position)

    trans = trans + a_center

    b = a - 0.15
    d = c - 0.25

    resolution = 360

    u = np.linspace(sigma, c, resolution)
    uu = np.linspace(sigma, d, resolution)
    v = np.linspace(0, 2 * np.pi, resolution)
    U, V = np.meshgrid(u, v)
    UU, VV = np.meshgrid(uu, v)

    X = U
    F = a * np.sqrt(1 - (U ** 2 / c ** 2))
    Y = F * np.cos(V)
    Z = F * np.sin(V)

    XX = UU
    G = b * np.sqrt(1 - (UU ** 2 / d ** 2))
    YY = G * np.cos(VV)
    ZZ = G * np.sin(VV)

    epicard_point = a * np.sqrt(1 - (sigma ** 2 / c ** 2))
    endocard_point = b * np.sqrt(1 - (sigma ** 2 / d ** 2))

    curvature = 10
    vv = np.linspace(endocard_point, epicard_point, resolution)

    VV, UV = np.meshgrid(vv, v)

    H = curvature * (VV - epicard_point) * (VV - endocard_point) + sigma

    XXX = H
    YYY = VV * np.cos(UV)
    ZZZ = VV * np.sin(UV)

    # rotation with arbitrary axle
    t = np.transpose(np.array([X, Y, Z]), (1, 2, 0))
    X, Y, Z = np.transpose(np.dot(t, rot_mx), (2, 0, 1))

    t = np.transpose(np.array([XX, YY, ZZ]), (1, 2, 0))
    XX, YY, ZZ = np.transpose(np.dot(t, rot_mx), (2, 0, 1))

    t = np.transpose(np.array([XXX, YYY, ZZZ]), (1, 2, 0))
    XXX, YYY, ZZZ = np.transpose(np.dot(t, rot_mx), (2, 0, 1))

    # scaling and translation
    X = np.exp(scale) * X + trans[0]
    Y = np.exp(scale) * Y + trans[1]
    Z = np.exp(scale) * Z + trans[2]

    XX = np.exp(scale) * XX + trans[0]
    YY = np.exp(scale) * YY + trans[1]
    ZZ = np.exp(scale) * ZZ + trans[2]

    XXX = np.exp(scale) * XXX + trans[0]
    YYY = np.exp(scale) * YYY + trans[1]
    ZZZ = np.exp(scale) * ZZZ + trans[2]

    lv_indicator_function = np.zeros(a_volume.shape)

    lv_indicator_function[X[:, :].astype(int), Y[:, :].astype(int), Z[:, :].astype(int)] = 1
    lv_indicator_function[XX[:, :].astype(int), YY[:, :].astype(int), ZZ[:, :].astype(int)] = 1
    lv_indicator_function[XXX[:, :].astype(int), YYY[:, :].astype(int), ZZZ[:, :].astype(int)] = 1
    lv_indicator_function = ndimage.binary_fill_holes(lv_indicator_function).astype(calc_type)

    if a_plot:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_axis_off()

        ax.plot_surface(X, Y, Z, alpha=0.7, color='green', rstride=6, cstride=12)
        ax.plot_surface(XX, YY, ZZ, alpha=0.7, color='green', rstride=6, cstride=12)
        ax.plot_surface(XXX, YYY, ZZZ, alpha=0.7, color='green', rstride=6, cstride=12)

        # fig = plt.show()

        image_name = 'left_ventricle' + str(a) + '.svg'
        image_format = 'svg'
        fig.savefig(image_name, format=image_format, dpi=1200)
        # ax.set_title('Left ventricle model')

    if a_plot_profile:
        ax = plt.subplot(111)

        ax.plot(a * np.sqrt(1 - (uu ** 2 / c ** 2)), color='blue')
        ax.plot(b * np.sqrt(1 - (u ** 2 / d ** 2)), color='red')

        h = curvature * (vv - epicard_point) * (vv - endocard_point)
        ax.plot(h, vv, color='green')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.show()

    return lv_indicator_function


def lv_cost_fn_vjp(a_volume, u, a_angle, a_trans, a_scale):
    """
    Manually calculated derivative (grad) of the object function. It is based on the paper [1]. 

    Args:
        a_volume (N, M, K): array_like
                  Only used for size of the indicator function generation
        
        u (N, M, K): array_like
           The volume image, scalar field of the reconstructed left ventricle
        
        a_angle (scalar): scalar
                 Parameter to handle the rotation of the left ventricle indicator rotation
        
        a_trans (scalar): scalar
                 Parameter expressing the translation of the model inside the volume set

        a_scale (scalar): scalar
                 Parameter to express the scalability of the scalability of the model

    Returns:
        rotation_grad (3): array_like
                       Gradient of the objective function, partial with the rotation param
        
        translation_grad (3): array_like
                          Gradient of the objective function, partial with the translation param
        
        scaling_grad (scalar): scalar
                      Gradient of objective function, partial with the scaling parameter


    [1] https://www.researchgate.net/profile/Niels-Overgaard-2/publication/221089710_Pose_Invariant_Shape_Prior_Segmentation_Using_Continuous_Cuts_and_Gradient_Descent_on_Lie_Groups/links/5d480f5aa6fdcc370a7c70d4/Pose-Invariant-Shape-Prior-Segmentation-Using-Continuous-Cuts-and-Gradient-Descent-on-Lie-Groups.pdf
    """
    rot_mx = rotation_mx(a_angle)

    params = [rotation_mx(a_angle), a_trans, a_scale]

    left_ventricle = signal.fftconvolve(lv_indicator(a_volume, [1, 2, -1], params), filter_kernel, mode='same')

    grad = np.transpose(np.asarray(np.gradient(left_ventricle)), (1, 2, 3, 0))

    rotation_clockwise = np.zeros([3, 3])
    rotation_clockwise[0, 2] = 1
    rotation_clockwise[1, 1] = -1
    rotation_clockwise[2, 0] = 1

    rotation_grad = np.zeros(3)

    image_domain = np.transpose(np.asarray(np.indices(a_volume.shape)), (1, 2, 3, 0))

    inner_prod = grad * ((image_domain - a_trans) @ rotation_clockwise.T)

    for i in range(3):
        rotation_grad[i] = np.sum((left_ventricle - u) * inner_prod[:, :, :, i]) \
                           / la.norm(grad * np.abs(image_domain - a_trans)) ** 2

    translation_grad = np.zeros(3)
    for i in range(3):
        translation_grad[i] = -np.sum((left_ventricle - u) * grad[:, :, :, i]) \
                              / la.norm(grad) ** 2

    scaling_grad = -np.sum((left_ventricle - u) * np.dot(grad, a_trans)) \
                   / la.norm(grad * np.abs(image_domain - a_trans)) ** 2

    return rotation_grad, translation_grad, scaling_grad


class arm:
    volume = []

    def __init__(self):
        """
        Initializing the ARM class.
        """
        self.volume = None

    def segment_left_ventricle(self, a_volume, a_opt_params, a_algo_params, a_plot=False, a_save_plot=False):
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
        par_lambda, par_nu, c_zero, c_one, b_zero, b_one = a_algo_params.values()

        angle = np.zeros(len(a_volume.shape))
        translation = np.zeros(len(a_volume.shape))
        scale = 1

        norm_epsilon = 0.001

        # mentioned in the paper
        b_zero = c_zero
        b_one = c_one

        self.volume = a_volume

        rows, cols, height = a_volume.shape

        # initialization for CMF
        if a_volume.dtype == np.int32:
            a_volume = a_volume.astype(calc_type)

        im_size = a_volume.size

        assert (rows * cols * height == im_size)

        alpha = 2 / (par_lambda + par_nu)

        lv_params = [1, 2, -1]

        f_zero = lv_indicator(a_volume, lv_params)
        f_one = f_zero

        im_eff = (par_lambda / (par_lambda + par_nu)) * a_volume + (par_nu / (par_lambda + par_nu)) \
                 * (b_zero * (1 - f_one) + b_one * f_one)

        Cs = (im_eff - c_zero) ** 2
        Ct = (im_eff - c_one) ** 2

        u = np.where(Cs >= Ct, 1, 0).astype(calc_type)

        ps = np.minimum(Cs, Ct)
        pt = ps

        pp_y = np.zeros((rows, cols + 1, height), dtype=calc_type)
        pp_x = np.zeros((rows + 1, cols, height), dtype=calc_type)
        pp_z = np.zeros((rows, cols, height + 1), dtype=calc_type)
        div_p = np.zeros((rows, cols, height), dtype=calc_type)

        cmf_iter = 3
        err_iter = np.zeros(cmf_iter * num_iter, dtype=calc_type)

        if a_plot is True:
            plt.ion()

            figure, axis = plt.subplots(2, 2)
            figure.tight_layout()
            
            slice_num = np.int32(a_volume.shape[0] / 2)

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

                gk = np.sqrt(squares * .5)
                gk = (gk <= alpha) + np.logical_not(gk <= alpha) * (gk / alpha)
                gk = 1 / gk

                pp_y[:, 1:-1, :] = (.5 * (gk[:, 1:, :] + gk[:, :-1, :])) * (pp_y[:, 1:-1, :])
                pp_x[1:-1, :, :] = (.5 * (gk[1:, :, :] + gk[:-1, :, :])) * (pp_x[1:-1, :, :])
                pp_z[:, :, 1:-1] = (.5 * (gk[:, :, 1:] + gk[:, :, :-1])) * (pp_z[:, :, 1:-1])

                div_p = pp_y[:, 1:, :] - pp_y[:, :-1, :]
                div_p += pp_x[1:, :, :] - pp_x[:-1, :, :]
                div_p += pp_z[:, :, 1:] - pp_z[:, :, :-1]

                # update the source flow ps
                pts = div_p + pt - u / gamma + 1 / gamma
                ps = np.minimum(pts, Cs)

                # update the sink flow pt
                pts = -div_p + ps + u / gamma
                pt = np.minimum(pts, Ct)

                u_error = gamma * (div_p - ps + pt)
                u -= u_error

                u_error_normed = np.sum(np.abs(u_error)) / im_size
                err_iter[cmf_iter * i + j] = u_error_normed

                if a_plot is True:
                    plot_obj_opt.set_data(u[slice_num, :, :])
                    axis[1, 0].plot(err_iter[0: cmf_iter * i + j])
                    plt.draw()

            c_zero = np.sum((1 - u) * im_eff) / np.sum(1 - u)
            c_one = np.sum(u * im_eff) / (np.sum(u))

            im_mod = c_zero * (1 - u) + c_one * u

            b_zero = np.sum((1 - f_one) * im_mod) / (la.norm(1 - f_one + norm_epsilon) ** 2)
            b_one = np.sum(f_one * im_mod) / (la.norm(f_one + norm_epsilon) ** 2)

            rotation_grad, translation_grad, scaling_grad = lv_cost_fn_vjp(a_volume, u, angle, translation, scale)

            epsilon = 10
            angle = angle - epsilon * rotation_grad
            translation = translation + epsilon * translation_grad
            scale = scale + epsilon * scaling_grad

            rot_mx = rotation_mx(angle)
            opt_transformation = [rot_mx, translation, scale]

            f_one = lv_indicator(a_volume, [1, 2, -1], opt_transformation)

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

            # next iter
