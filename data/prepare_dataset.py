import os
import json
import argparse
import glob
import random


def get_split(image_pairs, split_percentages):

    if ':' in split_percentages:
        num_of_images = len(image_pairs)
        sum = 0
        for perc in split_percentages.split(':'):
            sum += int(perc)
        if not sum == 100:
            print("ERROR: Sum of the split percentages is not 100")
            exit(1)
        if len(split_percentages.split(':')) == 3:
            test_part = int(float(split_percentages.split(':')[2]) / 100. * num_of_images)
            val_part = int(float(split_percentages.split(':')[1]) / 100. * num_of_images)
            train_part = num_of_images - (test_part + val_part)
            return [image_pairs[:train_part],
                    image_pairs[train_part:(train_part+val_part)],
                    image_pairs[(train_part+val_part):]]
        elif len(split_percentages.split(':')) == 2:
            val_part = int(float(split_percentages.split(':')[1]) / 100. * num_of_images)
            train_part = num_of_images - val_part
            return [image_pairs[:train_part], image_pairs[train_part:]]

    elif split_percentages == "100":
        return [image_pairs]
    else:
        print("Invalid --split_percentages argument. See the documentation for details")
        exit()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str, default='',
                        help="Path to images")
    parser.add_argument("--out_json", type=str, default='',
                        help="Path to output json file or output directory for more than one file")
    parser.add_argument("--split_percentages", type=str, default="75:25",
                        help="Split percentages in the form <train_percentage> "
                             "or <train_percentage>:<val_percentage> or"
                             "<train_percentage>:<val_percentage>:<test_percentage>")
    args = parser.parse_args()

    image_pairs = []
    all_dirs = os.listdir(args.images_path)
    for num, directory in enumerate(all_dirs):
        print(f"Progress: {num+1}/{len(all_dirs)} | {directory}")
        dir_path = os.path.join(args.images_path, directory)
        images_02 = glob.glob(os.path.join(dir_path, "*/image_02/data/*.png"))
        images_03 = glob.glob(os.path.join(dir_path, "*/image_03/data/*.png"))
        images_02.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
        images_03.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for image2, image3 in zip(images_02, images_03):
            # image2 = image2.replace("/", "\\").replace("\\\\", "\\")
            # image3 = image3.replace("/", "\\").replace("\\\\", "\\")
            image_pairs.append([image2, image3])

    random.shuffle(image_pairs)
    splits = get_split(image_pairs, args.split_percentages)

    if len(splits) == 1:
        with open(args.out_json, 'w') as outfile:
            json.dump(splits[0], outfile, indent=2)
    else:
        if not os.path.isdir(args.out_json):
            os.mkdir(args.out_json)
        for num, split in enumerate(splits):
            if num == 0:
                train_file = os.path.join(args.out_json, "train.json")
                with open(train_file, 'w') as outfile:
                    json.dump(split, outfile, indent=2)
            elif num == 1:
                val_file = os.path.join(args.out_json, "val.json")
                with open(val_file, 'w') as outfile:
                    json.dump(split, outfile, indent=2)
            else:
                test_file = os.path.join(args.out_json, "test.json")
                with open(test_file, 'w') as outfile:
                    json.dump(split, outfile, indent=2)
