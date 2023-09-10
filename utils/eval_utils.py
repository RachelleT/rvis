import itertools

import numpy as np
import cv2 as cv
import SimpleITK as sitk
import scipy.ndimage
from scipy.ndimage import map_coordinates
from utils import metrics


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Taken from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Useful for monitoring current value and running average over iterations.
    """

    def __init__(self):
        self.reset()
        self.values = np.array([])
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.values = np.array([])
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.values = np.append(self.values, val)
        self.std = np.std(self.values)


def compute_tre(fix_lms, mov_lms, spacing_fix, spacing_mov, disp=None, fix_lms_warped=None):
    if disp is not None:
        fix_lms_disp_x = map_coordinates(disp[:, :, :, 0], fix_lms.transpose())
        fix_lms_disp_y = map_coordinates(disp[:, :, :, 1], fix_lms.transpose())
        fix_lms_disp_z = map_coordinates(disp[:, :, :, 2], fix_lms.transpose())
        fix_lms_disp = np.array((fix_lms_disp_x, fix_lms_disp_y, fix_lms_disp_z)).transpose()

        fix_lms_warped = fix_lms + fix_lms_disp

    return np.linalg.norm((fix_lms_warped - mov_lms) * spacing_mov, axis=1)


def compute_landmark_accuracy(landmarks_pred, landmarks_gt, voxel_size):
    landmarks_pred = np.round(landmarks_pred)
    landmarks_gt = np.round(landmarks_gt)

    difference = landmarks_pred - landmarks_gt
    difference = np.abs(difference)
    difference = difference * voxel_size

    means = np.mean(difference, 0)
    stds = np.std(difference, 0)

    difference = np.square(difference)
    difference = np.sum(difference, 1)
    difference = np.sqrt(difference)

    means = np.append(means, np.mean(difference))
    stds = np.append(stds, np.std(difference))

    means = np.round(means, 2)
    stds = np.round(stds, 2)

    means = means[::-1]
    stds = stds[::-1]

    return means, stds


def compute_dice(fixed, moving_warped, labels):
    dice = []
    for i in labels:
        if ((fixed == i).sum() == 0) or ((moving_warped == i).sum() == 0):
            dice.append(np.NAN)
        else:
            dice.append(metrics.compute_dice_coefficient((fixed == i), (moving_warped == i)))
    mean_dice = np.nanmean(dice)
    return mean_dice, dice


def compute_hd95(fixed, moving_warped, labels):
    hd95 = []
    for i in labels:
        if ((fixed == i).sum() == 0) or ((moving_warped == i).sum() == 0):
            hd95.append(np.NAN)
        else:
            hd95.append(metrics.compute_robust_hausdorff(
                metrics.compute_surface_distances((fixed == i), (moving_warped == i), np.ones(3)), 95.))
    mean_hd95 = np.nanmean(hd95)
    return mean_hd95, hd95


def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape

    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)

    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (
            jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) - \
             jacobian[1, 0, :, :, :] * (
                     jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :,
                                                                                                   :, :]) + \
             jacobian[2, 0, :, :, :] * (
                     jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :,
                                                                                                   :, :])

    return jacdet


def binary_image(arr, threshold=0.8):
    total = np.sum(arr)
    if total > threshold:
        return True  # binary
    else:
        return False


def mask_image_multiply(mask, image):
    components_per_pixel = image.GetNumberOfComponentsPerPixel()
    if components_per_pixel == 1:
        return mask * image
    else:
        return sitk.Compose(
            [
                mask * sitk.VectorIndexSelectionCast(image, channel)
                for channel in range(components_per_pixel)
            ]
        )


def alpha_blend(image1, image2, alpha, mask1=None, mask2=None):
    """
    Alaph blend two images, pixels can be scalars or vectors.
    The alpha blending factor can be either a scalar or an image whose
    pixel type is sitkFloat32 and values are in [0,1].
    The region that is alpha blended is controled by the given masks.
    """

    if not mask1:
        mask1 = sitk.Image(image1.GetSize(), sitk.sitkFloat32) + 1.0
        mask1.CopyInformation(image1)
    else:
        mask1 = sitk.Cast(mask1, sitk.sitkFloat32)
    if not mask2:
        mask2 = sitk.Image(image2.GetSize(), sitk.sitkFloat32) + 1
        mask2.CopyInformation(image2)
    else:
        mask2 = sitk.Cast(mask2, sitk.sitkFloat32)
    # if we received a scalar, convert it to an image
    if type(alpha) != sitk.SimpleITK.Image:
        alpha = sitk.Image(image1.GetSize(), sitk.sitkFloat32) + alpha
        alpha.CopyInformation(image1)
    components_per_pixel = image1.GetNumberOfComponentsPerPixel()
    if components_per_pixel > 1:
        img1 = sitk.Cast(image1, sitk.sitkVectorFloat32)
        img2 = sitk.Cast(image2, sitk.sitkVectorFloat32)
    else:
        img1 = sitk.Cast(image1, sitk.sitkFloat32)
        img2 = sitk.Cast(image2, sitk.sitkFloat32)

    intersection_mask = mask1 * mask2

    intersection_image = mask_image_multiply(
        alpha * intersection_mask, img1
    ) + mask_image_multiply((1 - alpha) * intersection_mask, img2)
    return (
            intersection_image
            + mask_image_multiply(mask2 - intersection_mask, img2)
            + mask_image_multiply(mask1 - intersection_mask, img1)
    )


def make_isotropic(
        image,
        interpolator=sitk.sitkLinear,
        spacing=None,
        default_value=0,
        standardize_axes=False,
    ):
    """
    Many file formats (e.g. jpg, png,...) expect the pixels to be isotropic, same
    spacing for all axes. Saving non-isotropic data in these formats will result in
    distorted images. This function makes an image isotropic via resampling, if needed.
    Args:
        image (SimpleITK.Image): Input image.
        interpolator: By default the function uses a linear interpolator. For
                      label images one should use the sitkNearestNeighbor interpolator
                      so as not to introduce non-existant labels.
        spacing (float): Desired spacing. If none given then use the smallest spacing from
                         the original image.
        default_value (image.GetPixelID): Desired pixel value for resampled points that fall
                                          outside the original image (e.g. HU value for air, -1000,
                                          when image is CT).
        standardize_axes (bool): If the original image axes were not the standard ones, i.e. non
                                 identity cosine matrix, we may want to resample it to have standard
                                 axes. To do that, set this paramter to True.
    Returns:
        SimpleITK.Image with isotropic spacing which occupies the same region in space as
        the input image.
    """
    original_spacing = image.GetSpacing()
    # Image is already isotropic, just return a copy.
    if all(spc == original_spacing[0] for spc in original_spacing):
        return sitk.Image(image)
    # Make image isotropic via resampling.
    original_size = image.GetSize()
    if spacing is None:
        spacing = min(original_spacing)
    new_spacing = [spacing] * image.GetDimension()
    new_size = [
        int(round(osz * ospc / spacing))
        for osz, ospc in zip(original_size, original_spacing)
    ]
    new_direction = image.GetDirection()
    new_origin = image.GetOrigin()
    # Only need to standardize axes if user requested and the original
    # axes were not standard.
    if standardize_axes and not np.array_equal(
            np.array(new_direction), np.identity(image.GetDimension()).ravel()
    ):
        new_direction = np.identity(image.GetDimension()).ravel()
        # Compute bounding box for the original, non standard axes image.
        boundary_points = []
        for boundary_index in list(
                itertools.product(*zip([0] * image.GetDimension(), image.GetSize()))
        ):
            boundary_points.append(image.TransformIndexToPhysicalPoint(boundary_index))
        max_coords = np.max(boundary_points, axis=0)
        min_coords = np.min(boundary_points, axis=0)
        new_origin = min_coords
        new_size = (((max_coords - min_coords) / spacing).round().astype(int)).tolist()
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        new_origin,
        new_spacing,
        new_direction,
        default_value,
        image.GetPixelID(),
    )


def overlayMask(image, mask, mColormap, alphaNumber):
    """ apply mask to image

        Parameters:
        image: sitk object
        mask: sitk object
        mColormap: string
        alphaNumber: float

        Returns: sitk object

    """
    # Dictionary with functions mapping a scalar image to a three component vector image
    image_mappings = {'grey': lambda x: sitk.ScalarToRGBColormap(x, sitk.ScalarToRGBColormapImageFilter.Grey),
                      'Jet': lambda x: sitk.ScalarToRGBColormap(x, sitk.ScalarToRGBColormapImageFilter.Jet),
                      'Hot': lambda x: sitk.ScalarToRGBColormap(x, sitk.ScalarToRGBColormapImageFilter.Hot)}

    image = make_isotropic(image, interpolator=sitk.sitkLinear)
    segmentation = make_isotropic(mask, interpolator=sitk.sitkNearestNeighbor)

    # Convert image to sitkUInt8 after rescaling, color image formats only work for [0,255]
    image_255 = sitk.Cast(sitk.RescaleIntensity(image, 0, 255), sitk.sitkUInt8)

    colormap = 'grey'
    vec_image = image_mappings[colormap](image_255)
    vec_segmentation = image_mappings[mColormap](segmentation)
    vec_combined = sitk.Cast(alpha_blend(image1=vec_image, image2=vec_segmentation, alpha=alphaNumber, mask2=segmentation == 1),
                             sitk.sitkVectorUInt8)
    return vec_combined
