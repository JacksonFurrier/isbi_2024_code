import numpy as np
import astra
import scipy.sparse.linalg
from skimage.transform import radon, iradon
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale

vol_geometry = None
proj_geometry = None
astra_wrapper = None


def init_astra_vars(a_vol_geom, a_proj_geom, a_astra_wrap):
    """
    Astra variable initialization function, which set 
    global variables. 

    TODO: Astra reconstruction is completed in the tests,
    integrate some of them here.

    Args:
        a_vol_geom (N, M, K): array_like
                Desired volume geometry for the reconstruction
        a_proj_geom (M, K): array_like
                Projection geometry settings, where it is usually 2D
        a_astra_wrap (function): lambda
                Astra wrapper function
    Returns:
        None
    """
    global vol_geometry, proj_geometry, astra_wrapper
    vol_geometry = a_vol_geom
    proj_geometry = a_proj_geom
    astra_wrapper = a_astra_wrap


# FP/BP wrapper class
class astra_wrap(object):
    """
    Astra wrapper class that has all the information for forward and backprojection as well.

    Args:
        object (object): object
                Simple object that is being passed down here by the astra environment
    
    Returns:
        The desired astra object TODO: decide how to handle these
    """
    def __init__(self, a_proj_geom, a_vol_geom):
        self.proj_id = astra.create_projector('cuda', a_proj_geom, a_vol_geom)
        self.shape = (a_proj_geom['DetectorCount'] * len(a_proj_geom['ProjectionAngles']),
                      a_vol_geom['GridColCount'] * a_vol_geom['GridRowCount'])
        self.dtype = np.float

    def matvec(self, v):
        global vol_geometry
        sid, s = astra.create_sino(np.reshape(v, (vol_geometry['GridRowCount'], vol_geometry['GridColCount'])),
                                   self.proj_id)
        astra.data2d.delete(sid)
        return s.ravel()

    def rmatvec(self, v):
        global proj_geometry
        bid, b = astra.create_backprojection(
            np.reshape(v, (len(proj_geometry['ProjectionAngles']), proj_geometry['DetectorCount'],)), self.proj_id)
        astra.data2d.delete(bid)
        return b.ravel()


def backward_projector(a_mode='basic'):
    if a_mode == 'astra':
        backward_proj = lambda a_projs: astra_backward(a_projs)
    if a_mode == 'basic':
        backward_proj = lambda a_projs: inverse_radon_transform(a_projs)

    return backward_proj


def astra_backward(a_projs):
    global vol_geometry, proj_geometry
    vol_dim = np.max(a_projs.shape)
    volume = np.zeros([vol_dim, vol_dim, vol_dim])
    sinograms = np.transpose(np.zeros(a_projs.shape), [1, 0, 2])

    for i in range(0, vol_dim):
        sinograms[i, :, :] = a_projs[:, i, :]
        b = sinograms[i, :, :].ravel()
        temp = scipy.sparse.linalg.lsqr(astra_wrapper, b, atol=1e-4, btol=1e-4, iter_lim=20)
        volume[i, :, :] = np.reshape(temp[0], [vol_geometry['GridRowCount'], vol_geometry['GridColCount']])
    return volume


def inverse_radon_transform(a_proj_frames):
    size = np.zeros(len(a_proj_frames.shape), dtype=np.int32)

    if np.min(a_proj_frames.shape) != np.max(a_proj_frames.shape):
        size[:] = np.max(a_proj_frames.shape)
    else:
        size = a_proj_frames.shape

    volume = np.zeros(size)
    sinograms = np.transpose(np.zeros(a_proj_frames.shape), [1, 0, 2])

    if a_proj_frames.shape[0] is not a_proj_frames.shape[1]:
        num_slices = a_proj_frames.shape[1]
    else:
        num_slices = a_proj_frames.shape[0]

    # usual angle interval for cardiac scans, will be derived from the dicom file later
    proj_angle_int = np.linspace(0, 210, a_proj_frames.shape[0], endpoint=False)
    for i in range(0, num_slices):
        sinograms[i, :, :] = a_proj_frames[:, i, :]
        volume[i, :, :] = iradon(np.transpose(sinograms[i, :, :]), theta=proj_angle_int, filter_name='hamming')

    return volume


def forward_projector(a_mode='basic'):
    if a_mode == 'astra':
        forward_proj = lambda a_volume: astra_forward(a_volume)
    if a_mode == 'basic':
        forward_proj = lambda a_volume: radon_transform(a_volume)

    return forward_proj


# we will need some code to determine if we are able to run GPU reconstruction or not

def astra_forward(a_volume):
    global vol_geometry, proj_geometry, astra_wrapper
    singoram_id, sinogram = astra.create_sino(a_volume, astra_wrapper.proj_id)
    astra.data2d.delete(singoram_id)

    return sinogram


def radon_transform(a_volume):
    proj_frames = np.zeros(a_volume.shape)
    sinograms = np.zeros(a_volume.shape)

    num_slices = a_volume.shape[0]
    # usual angle interval for cardiac scans, will be derived from the dicom file later
    proj_angle_int = np.linspace(0, 210, max(a_volume[0, :, :].shape), endpoint=False)
    for i in range(0, num_slices):
        sinograms[i, :, :] = radon((a_volume[i, :, :]), proj_angle_int)
        proj_frames[:, i, :] = np.transpose(sinograms[i, :, :])

    return proj_frames
