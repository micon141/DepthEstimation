import os
import glob
import argparse
import json
import random
from tqdm import tqdm
import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from dataset import preprocess_image
from model import ResNet34Encoder, ResNet50Encoder, ResNet101Encoder, ResNet152Encoder
from utils import get_device
from eval import disp_to_depth_gt, disp_to_depth_pred, compute_errors


X_CROP = 15


def get_images_from_json(json_file):
    with open(json_file, 'r') as f:
        image_pairs = json.load(f)
    images_left = [image_pair[0] for image_pair in image_pairs]
    return images_left


def get_model(model_path, image_size, architecture, device):
    if architecture == "resnet50":
        model = ResNet50Encoder(image_size, 3).cuda(device)
    elif architecture == "resnet101":
        model = ResNet101Encoder(image_size, 3).cuda(device)
    elif architecture == "resnet152":
        model = ResNet152Encoder(image_size, 3).cuda(device)
    elif architecture == "resnet34":
        model = ResNet34Encoder(image_size, 3).cuda(device)
    else:
        raise NotImplementedError(f"Architecture {architecture} is not implemented. "
                                  f"See documentation for supported architectures")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model.cuda(device)


def inference_on_single_image(image, image_size, model, device):
    image = preprocess_image(image, image_size)
    image = torch.from_numpy(image).type(torch.FloatTensor).cuda(device)
    image = image.unsqueeze(0)
    with torch.no_grad():
        disparities = model(image)
    disp = disparities[0]
    return disp


def save_image_and_depth_map(image_path, output_dir, pred_disp, image_org):
    pred_disp = pred_disp[:, X_CROP:pred_disp.shape[1] - X_CROP]
    image_org = image_org[:, X_CROP:image_org.shape[1] - X_CROP, :]
    out_depth_map_path = os.path.join(output_dir, image_path.split('/')[-1].split('.')[0] + "_depth_map.jpg")
    plt.imsave(out_depth_map_path, pred_disp, cmap="plasma")
    out_image_path = os.path.join(output_dir, image_path.split('/')[-1])
    cv.imwrite(out_image_path, image_org)


def calc_metrics(pred_disp, gt_disp, min_depth, max_depth):
    gt_depth = disp_to_depth_gt(ground_truth)
    pred_disp[pred_disp < min_depth] = min_depth
    pred_disp[pred_disp > max_depth] = max_depth
    gt_depth[gt_depth < min_depth] = min_depth
    gt_depth[gt_depth > max_depth] = max_depth
    mask = gt_disp > 0

    metrics = compute_errors(gt_depth[mask], pred_disp[mask], mask)
    return metrics


def get_depth(image, image_size, model, device):
    disparity = inference_on_single_image(image, image_size, model, device)
    disparity = disparity[0, 1, :, :].cpu().numpy()
    depth_map, pred_disp = disp_to_depth_pred(disparity, (image.shape[1], image.shape[0]))
    return depth_map, pred_disp


def read_gt(gt_path):
    gt_disp = cv.imread(gt_path, -1)
    gt_disp = gt_disp.astype(np.float32) / 256
    return gt_disp


def concateante_image_depth(image_org, pred_disp, image_size):
    image_org = cv.resize(image_org, tuple(image_size))
    pred_disp = cv.resize(pred_disp, tuple(image_size))
    pred_disp = cv.cvtColor(pred_disp, cv.COLOR_GRAY2RGB)
    pred_disp = pred_disp.astype(np.uint8)
    image_to_write = np.concatenate((image_org, pred_disp), axis=0)
    image_to_write = image_to_write[:, X_CROP:image_size[0] - X_CROP, :]
    return image_to_write


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default='',
                        help="Path to images or video")
    parser.add_argument("--architecture", type=str, default='',
                        help="Model architecture that is used during training")
    parser.add_argument("--model_path", type=str, default='',
                        help="Path to the model")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU id")
    parser.add_argument("--image_size", type=str, default="512x320",
                        help="Input size of the network in the format WxH")
    parser.add_argument("--output", type=str, default='',
                        help="Directory for storing images or output video file")
    parser.add_argument("--min-depth", type=float, default=0.001)
    parser.add_argument("--max-depth", type=float, default=80)
    args = parser.parse_args()

    image_size = [int(args.image_size.split('x')[0]), int(args.image_size.split('x')[1])]
    device = get_device(args.device)
    model = get_model(args.model_path, image_size, args.architecture, device)

    print("Obtaining depth map...")
    if args.source.endswith(".jpg") or args.source.endswith(".png"):
        image_org = cv.imread(args.source)
        pred_depth, pred_disp = get_depth(image_org.copy(), image_size, model, device)
        if args.output:
            save_image_and_depth_map(args.source, args.output, pred_disp, image_org)
        ground_truth = [read_gt(args.source)]
        calc_metrics(pred_disp, ground_truth, args.min_depth, args.max_depth)
    elif args.source.endswith(".json"):
        image_paths = get_images_from_json(args.source)
        random.shuffle(image_paths)
        num_images = len(image_paths)
        metrics = np.zeros((8, num_images), np.float32)
        num = 0
        for image_path in tqdm(image_paths):
            gt_path = image_path.replace("images", "groundtruth")
            if not os.path.exists(gt_path):
                continue
            image_org = cv.imread(image_path)
            pred_depth, pred_disp = get_depth(image_org.copy(), image_size, model, device)
            ground_truth = read_gt(gt_path)
            gt_depth = disp_to_depth_gt(ground_truth)
            if args.output:
                save_image_and_depth_map(image_path, args.output, pred_disp, image_org)
            met = calc_metrics(pred_disp, ground_truth, args.min_depth, args.max_depth)
            num += 1
            metrics[:, num] = met
        metrics = np.mean(metrics, axis=1)
        print(f"abs_rel:  {metrics[0]}\nsq_rel:   {metrics[1]}\nrmse:     {metrics[2]}\nlog_rmse: {metrics[3]}\n"
              f"a1:       {metrics[4]}\na2:       {metrics[5]}\na3:       {metrics[6]}\nd1:       {metrics[7]}")
    elif os.path.isdir(args.source):
        images = glob.glob(os.path.join(args.source, "image_02/data/", '*'))
        images.sort(key=lambda x: int(x.split('/')[-1].split(".png")[0].split(".jpg")[0]))
        out = None
        if not os.path.isdir(args.output):
            out = cv.VideoWriter(args.output, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                 (image_size[0] - 2*X_CROP, 2*image_size[1]))
        num_images = len(images)
        for image_path in tqdm(images):
            image_org = cv.imread(image_path)
            depth_map, pred_disp = get_depth(image_org.copy(), image_size, model, device)
            if args.output.endswith("avi") or args.output.endswith("mp4"):
                pred_disp = pred_disp / pred_disp.max() * 255.
                image_to_write = concateante_image_depth(image_org, pred_disp, image_size)
                out.write(image_to_write)
            else:
                if not os.path.isdir(args.output):
                    os.mkdir(args.output)
                save_image_and_depth_map(image_path, args.output, pred_disp, image_org)

        if out:
            out.release()
    else:
        print("Not valid --source file")
        exit()
