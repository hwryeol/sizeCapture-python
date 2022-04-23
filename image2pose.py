import torch
import cv2
import math
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
import pose2mesh


def setup_cfg(config_file):
    confidence_threshold = 0.5
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list([])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg

def getImageJoint(inputImgPath):
    config = 'detectron2/configs/quick_schedules/keypoint_rcnn_R_50_FPN_inference_acc_test.yaml'
    cfg = setup_cfg(config)
    predictor = DefaultPredictor(cfg)

    image = cv2.imread(inputImgPath, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    predictions = predictor(image)
    tensor_data = predictions['instances'].pred_keypoints
    return tensor_data




