import numpy as np
import cv2 as cv


# Values for focal length for different input shape
WIDTH_TO_FOCAL = dict()
WIDTH_TO_FOCAL[1242] = 721.5377
WIDTH_TO_FOCAL[1241] = 718.856
WIDTH_TO_FOCAL[1224] = 707.0493
WIDTH_TO_FOCAL[1226] = 708.0643
WIDTH_TO_FOCAL[1238] = 718.3351


def disp_to_depth_gt(disp):
    """
    :param disp: Disparity tensor
    :return: Converted depth map
    """
    height, width = disp.shape
    mask = disp > 0
    depth = WIDTH_TO_FOCAL[width] * 0.54 / (disp + (1.0 - mask))
    return depth


def disp_to_depth_pred(disp, input_shape):
    """
    :param disp: Disparity tensor
    :param input_shape: Shape of the input image
    :return: Converted depth map
    """
    width, height = input_shape
    disp = width * cv.resize(disp, input_shape, interpolation=cv.INTER_LINEAR)
    depth = WIDTH_TO_FOCAL[width] * 0.54 / disp
    return depth, disp


def compute_errors(gt, pred, mask):
    """
    :param gt: Ground truth
    :param pred: Predictions
    :param mask: Mask
    :return:
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    disp_diff = np.abs(gt - pred)
    bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt) >= 0.05)
    d1_all = 100.0 * bad_pixels.sum() / mask.sum()

    return [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, d1_all]
