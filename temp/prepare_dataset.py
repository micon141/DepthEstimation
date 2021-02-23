import os
import json
import argparse
import glob

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str, default='',
                        help="Path to images")
    parser.add_argument("--out_json", type=str, default='',
                        help="Path to output json file")
    args = parser.parse_args()

    image_pairs = []
    for directory in os.listdir(args.images_path):
        dir_path = os.path.join(args.images_path, directory)
        images_02 = glob.glob(os.path.join(dir_path, "*/image_02/data/*.png"))
        images_03 = glob.glob(os.path.join(dir_path, "*/image_03/data/*.png"))
        for image2, image3 in zip(images_02, images_03):
            image2 = image2.replace("/", "\\").replace("\\\\", "\\")
            image3 = image3.replace("/", "\\").replace("\\\\", "\\")
            image_pairs.append([image2, image3])

    with open(args.out_json, 'w') as outfile:
        json.dump(image_pairs, outfile, indent=2)
