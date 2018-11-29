#!/usr/bin/env python
import os
import cv2
import json
import torch
import pprint
import argparse
import importlib
import numpy as np

import matplotlib
matplotlib.use("Agg")

from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
from test.detect import kp_decode
from test.detect import _rescale_dets
from tqdm import tqdm
from config import system_configs
from utils import crop_image, normalize_
from external.nms import soft_nms, soft_nms_merge

torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Test CornerNet")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=None, type=int)
    parser.add_argument("--split", dest="split",
                        help="which split to use",
                        default="validation", type=str)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
def test(db, split, testiter, debug=False, suffix=None):
    result_dir = system_configs.result_dir
    result_dir = os.path.join(result_dir, str(testiter), split)
    class_name = []
    for i in range(1, len(db._coco.cats)):
        # if db._coco.cats[i] is None:
        #     continue
        # else:
        ind = db._cat_ids[i]
        class_name.append(db._coco.cats[ind]['name'])
    if suffix is not None:
        result_dir = os.path.join(result_dir, suffix)

    make_dirs([result_dir])

    test_iter = system_configs.max_iter if testiter is None else testiter
    print("loading parameters at iteration: {}".format(test_iter))

    print("building neural network...")
    nnet = NetworkFactory(db)
    print("loading parameters...")
    nnet.load_params(test_iter)

    # test_file = "test.{}".format(db.data)
    # testing = importlib.import_module(test_file).testing

    nnet.cuda()
    nnet.eval_mode()

    debug_dir = os.path.join(result_dir, "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    if db.split != "trainval":
        db_inds = db.db_inds[:100] if debug else db.db_inds
    else:
        db_inds = db.db_inds[:100] if debug else db.db_inds[:5000]

    K = db.configs["top_k"]
    ae_threshold = db.configs["ae_threshold"]
    nms_kernel = db.configs["nms_kernel"]

    scales = db.configs["test_scales"]
    weight_exp = db.configs["weight_exp"]
    merge_bbox = db.configs["merge_bbox"]
    categories = db.configs["categories"]
    nms_threshold = db.configs["nms_threshold"]
    max_per_image = db.configs["max_per_image"]
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1,
        "exp_soft_nms": 2
    }[db.configs["nms_algorithm"]]

    img_name = os.listdir(db._image_dir)
    for i in range(0, len(img_name)):
        top_bboxes = {}
        # for ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
        db_ind = i + 1

        # image_id = db.image_ids(db_ind)
        image_id = img_name[i]
        image_file = db._image_dir + '/' + img_name[i]
        image = cv2.imread(image_file)

        height, width = image.shape[0:2]

        detections = []

        for scale in scales:
            new_height = int(height * scale)
            new_width = int(width * scale)
            new_center = np.array([new_height // 2, new_width // 2])

            inp_height = new_height | 127
            inp_width = new_width | 127

            images = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
            ratios = np.zeros((1, 2), dtype=np.float32)
            borders = np.zeros((1, 4), dtype=np.float32)
            sizes = np.zeros((1, 2), dtype=np.float32)

            out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
            height_ratio = out_height / inp_height
            width_ratio = out_width / inp_width

            resized_image = cv2.resize(image, (new_width, new_height))
            resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])

            resized_image = resized_image / 255.
            normalize_(resized_image, db.mean, db.std)

            images[0] = resized_image.transpose((2, 0, 1))
            borders[0] = border
            sizes[0] = [int(height * scale), int(width * scale)]
            ratios[0] = [height_ratio, width_ratio]

            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
            images = torch.from_numpy(images)
            dets = kp_decode(nnet, images, K, ae_threshold=ae_threshold, kernel=nms_kernel)
            dets = dets.reshape(2, -1, 8)
            dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
            dets = dets.reshape(1, -1, 8)

            _rescale_dets(dets, ratios, borders, sizes)
            dets[:, :, 0:4] /= scale
            detections.append(dets)

        detections = np.concatenate(detections, axis=1)

        classes = detections[..., -1]
        classes = classes[0]
        detections = detections[0]

        # reject detections with negative scores
        keep_inds = (detections[:, 4] > -1)
        detections = detections[keep_inds]
        classes = classes[keep_inds]

        top_bboxes[image_id] = {}
        for j in range(categories):
            keep_inds = (classes == j)
            top_bboxes[image_id][j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
            if merge_bbox:
                soft_nms_merge(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm,
                               weight_exp=weight_exp)
            else:
                soft_nms(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm)
            top_bboxes[image_id][j + 1] = top_bboxes[image_id][j + 1][:, 0:5]

        scores = np.hstack([
            top_bboxes[image_id][j][:, -1]
            for j in range(1, categories + 1)
        ])
        if len(scores) > max_per_image:
            kth = len(scores) - max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, categories + 1):
                keep_inds = (top_bboxes[image_id][j][:, -1] >= thresh)
                top_bboxes[image_id][j] = top_bboxes[image_id][j][keep_inds]

        # result_json = os.path.join(result_dir, "results.json")
        detections = db.convert_to_list(top_bboxes)
        print('demo for {}'.format(image_id))
        img = cv2.imread(image_file)
        box = []
        if detections is not None:
            for i in range(len(detections)):
                name = db._coco.cats[detections[i][1]]['name']  #db._coco.cats[ind]['name']
                confi = detections[i][-1]
                if confi <0.3:
                    continue
                for j in range(0, 4):
                    box.append(detections[i][j + 2])
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 1)
                # cv2.putText(img, name[0] + '  ' + '{:.3f}'.format(confi), (int(box[0]), int(box[1] - 10)),
                #             cv2.FONT_ITALIC, 1, (0, 0, 255), 1)
                while (box):
                    box.pop(-1)
        cv2.imshow('Detecting image...', img)
        # timer.total_time = 0
        if cv2.waitKey(3000) & 0xFF == ord('q'):
            break
        print(detections)

if __name__ == "__main__":
    args = parse_args()

    if args.suffix is None:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    else:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + "-{}.json".format(args.suffix))
    print("cfg_file: {}".format(cfg_file))

    with open(cfg_file, "r") as f:
        configs = json.load(f)

    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split = system_configs.val_split
    test_split = system_configs.test_split

    split = {
        "training": train_split,
        "validation": val_split,
        "testing": test_split
    }[args.split]

    print("loading all datasets...")
    dataset = system_configs.dataset
    print("split: {}".format(split))
    testing_db = datasets[dataset](configs["db"], split)

    print("system config...")
    pprint.pprint(system_configs.full)

    print("db config...")
    pprint.pprint(testing_db.configs)

    test(testing_db, args.split, args.testiter, args.debug, args.suffix)