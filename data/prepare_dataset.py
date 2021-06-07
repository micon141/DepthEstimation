import os
import json
import argparse
import glob
import random
from tqdm import tqdm


def get_splits(split_percentages, all_dirs):
    dirs_num = len(all_dirs)
    train_dirs, val_dirs, test_dirs = 0, 0, 0
    if len(split_percentages.split(':')) == 1:
        train_dirs = all_dirs
    elif len(split_percentages.split(':')) == 2:
        train_num = int(float(split_percentages.split(':')[0]) / 100. * dirs_num)
        train_dirs = all_dirs[:train_num]
        val_dirs = all_dirs[train_num:]
    elif len(split_percentages.split(':')) == 3:
        train_num = int(float(split_percentages.split(':')[0]) / 100. * dirs_num)
        val_num = int(float(split_percentages.split(':')[1]) / 100. * dirs_num)
        train_dirs = all_dirs[:train_num]
        val_dirs = all_dirs[:(train_num + val_num)]
        test_dirs = all_dirs[(train_num + val_num):]
    else:
        print("Invalid argument: split_percentages")
        exit(1)
    return train_dirs, val_dirs, test_dirs


def generate_json(dirs, json_path):
    image_pairs = []
    for directory in tqdm(dirs):
        images_02 = glob.glob(os.path.join(directory, "image_02/data/*.png"))
        images_03 = glob.glob(os.path.join(directory, "image_03/data/*.png"))
        images_02.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
        images_03.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for image2, image3 in zip(images_02, images_03):
            image_pairs.append([image2, image3])
    print(f"Total number of images: {len(image_pairs)}")
    with open(json_path, 'w') as f:
        json.dump(image_pairs, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str, default='',
                        help="Path to images")
    parser.add_argument("--train_json", type=str, default='',
                        help="Path to output json file - train data")
    parser.add_argument("--val_json", type=str, default='',
                        help="Path to output json file - validation data")
    parser.add_argument("--test_json", type=str, default='',
                        help="Path to output json file - test data")
    parser.add_argument("--split_percentages", type=str, default="75:25",
                        help="Split percentages in the form <train_percentage> "
                             "or <train_percentage>:<val_percentage> or"
                             "<train_percentage>:<val_percentage>:<test_percentage>")
    args = parser.parse_args()

    print("Splitting dataset...")
    all_dirs = glob.glob(os.path.join(args.images_path, "*/*"))
    all_dirs = [path for path in all_dirs if os.path.isdir(path)]
    random.shuffle(all_dirs)
    train_dirs, val_dirs, test_dirs = get_splits(args.split_percentages, all_dirs)

    print(f"Generating train dataset...\nNumber of directories: {len(train_dirs)}")
    generate_json(train_dirs, args.train_json)
    if val_dirs:
        print(f"Generating val dataset...\nNumber of directories: {len(val_dirs)}")
        generate_json(val_dirs, args.val_json)
    if test_dirs:
        print(f"Generating test dataset...\nNumber of directories: {len(test_dirs)}")
        generate_json(test_dirs, args.test_json)
