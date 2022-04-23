import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import colorsys
import math
import json
import argparse

import __init_path
import models
from core.config import cfg
from aug_utils import j2d_processing
from coord_utils import get_bbox, process_bbox
from funcs_utils import load_checkpoint, save_obj
from graph_utils import build_coarse_graphs
from renderer import Renderer
from vis import vis_2d_keypoints, vis_coco_skeleton
from _mano import MANO
from smpl import SMPL

import random


def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    x, y, w, h = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:, 3]
    cx, cy, h = x + w/2, y + h/2, h
    # cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam


def render(result, orig_height, orig_width, orig_img, mesh_face, color):
    pred_verts, pred_cam, bbox = result['mesh'], result['cam_param'][None, :], result['bbox'][None, :]

    orig_cam = convert_crop_cam_to_orig_img(
        cam=pred_cam,
        bbox=bbox,
        img_width=orig_width,
        img_height=orig_height
    )

    # Setup renderer for visualization
    renderer = Renderer(mesh_face, resolution=(orig_width, orig_height), orig_img=True, wireframe=False)
    renederd_img = renderer.render(
        orig_img,
        pred_verts,
        cam=orig_cam[0],
        color=color,
        mesh_filename=None,
        rotate=False
    )
    return renederd_img


def get_joint_setting(mesh_model, joint_category='human36'):
    joint_regressor, joint_num, skeleton, graph_L, graph_perm_reverse = None, None, None, None, None
    if joint_category == 'human36':
        joint_regressor = mesh_model.joint_regressor_h36m
        joint_num = 17
        skeleton = (
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
        (2, 3), (0, 4), (4, 5), (5, 6))
        flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        graph_Adj, graph_L, graph_perm,graph_perm_reverse = \
            build_coarse_graphs(mesh_model.face, joint_num, skeleton, flip_pairs, levels=9)
        model_chk_path = './experiment/pose2mesh_human36J_train_human36/final.pth.tar'

    elif joint_category == 'coco':
        joint_regressor = mesh_model.joint_regressor_coco
        joint_num = 19  # add pelvis and neck
        skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
            (13, 15),  # (5, 6), #(11, 12),
            (17, 11), (17, 12), (17, 18), (18, 5), (18, 6), (18, 0))
        flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        graph_Adj, graph_L, graph_perm, graph_perm_reverse = \
            build_coarse_graphs(mesh_model.face, joint_num, skeleton, flip_pairs, levels=9)
        model_chk_path = './experiment/pose2mesh_cocoJ_train_human36_coco_muco/final.pth.tar'
    else:
        raise NotImplementedError(f"{joint_category}: unknown joint set category")

    model = models.pose2mesh_net.get_model(joint_num, graph_L)
    checkpoint = load_checkpoint(load_dir=model_chk_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, joint_regressor, joint_num, skeleton, graph_L, graph_perm_reverse


def add_pelvis(joint_coord, joints_name):
    lhip_idx = joints_name.index('L_Hip')
    rhip_idx = joints_name.index('R_Hip')
    pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
    pelvis[2] = joint_coord[lhip_idx, 2] * joint_coord[rhip_idx, 2]  # confidence for pelvis
    pelvis = pelvis.reshape(1, 3)

    joint_coord = np.concatenate((joint_coord, pelvis))
    return joint_coord


def add_neck(joint_coord, joints_name):
    lshoulder_idx = joints_name.index('L_Shoulder')
    rshoulder_idx = joints_name.index('R_Shoulder')
    neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
    neck[2] = joint_coord[lshoulder_idx, 2] * joint_coord[rshoulder_idx, 2]  # confidence for neck
    neck = neck.reshape(1,3)

    joint_coord = np.concatenate((joint_coord, neck))
    return joint_coord


def optimize_cam_param(project_net, joint_input, crop_size, metadata):
    bbox = get_bbox(joint_input)
    bbox1 = process_bbox(bbox.copy(), aspect_ratio=1.0, scale=1.25)
    bbox2 = process_bbox(bbox.copy())
    proj_target_joint_img, trans = j2d_processing(joint_input.copy(), (crop_size, crop_size), bbox1, 0, 0, None)
    joint_img, _ = j2d_processing(joint_input.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]), bbox2, 0, 0, None)

    joint_img = joint_img[:, :2]
    joint_img /= np.array([[cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]]])
    mean, std = np.mean(joint_img, axis=0), np.std(joint_img, axis=0)
    joint_img = (joint_img.copy() - mean) / std
    joint_img = torch.Tensor(joint_img[None, :, :]).cuda()
    target_joint = torch.Tensor(proj_target_joint_img[None, :, :2]).cuda()

    # get optimization settings for projection
    criterion = nn.L1Loss()
    optimizer = optim.Adam(project_net.parameters(), lr=0.1)

    # estimate mesh, pose
    metadata['model'].eval()
    pred_mesh, _ = metadata['model'](joint_img)
    pred_mesh = pred_mesh[:, metadata['graph_perm_reverse'][:metadata['mesh_model'].face.max() + 1], :]
    pred_3d_joint = torch.matmul(metadata['joint_regressor'], pred_mesh)

    out = {}
    # assume batch=1
    project_net.train()
    for j in range(0, 1500):
        # projection
        pred_2d_joint = project_net(pred_3d_joint.detach())

        loss = criterion(pred_2d_joint, target_joint[:, :17, :])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if j == 500:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.05
        if j == 1000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001

    out['mesh'] = pred_mesh[0].detach().cpu().numpy()
    out['cam_param'] = project_net.cam_param[0].detach().cpu().numpy()
    out['bbox'] = bbox1

    out['target'] = proj_target_joint_img

    return out


def getRangeJoint(img,tensor_data,bbox):
    nose = (tensor_data[0][0],tensor_data[1][0])
    neck = (tensor_data[0][17],tensor_data[1][17])
    pelvis = (tensor_data[0][18],tensor_data[1][18])
    L_hip = (tensor_data[0][11],tensor_data[1][11])
    R_hip = (tensor_data[0][12],tensor_data[1][12])
    knee = (tensor_data[0][13],tensor_data[1][13])
    ankle = (tensor_data[0][15],tensor_data[1][15])
    L_Shoulder = (tensor_data[0][5],tensor_data[1][5])
    R_Shoulder = (tensor_data[0][6], tensor_data[1][6])
    L_Elbow = (tensor_data[0][7],tensor_data[1][7])
    L_Wrist = (tensor_data[0][9],tensor_data[1][9])
    L_ear = (tensor_data[0][3],tensor_data[1][3])

    nose2ear = math.sqrt(pow(nose[0]-L_ear[0],2) + pow(nose[1]-L_ear[1],2))


    A = math.sqrt(pow(neck[0]-nose[0],2) + pow(neck[1]-nose[1],2))
    B = math.sqrt(pow(neck[0]-pelvis[0],2) + pow(neck[1]-pelvis[1],2))
    C = math.sqrt(pow(knee[0]-L_hip[0],2) + pow(knee[1]-L_hip[1],2))
    D = math.sqrt(pow(ankle[0]-knee[0],2) + pow(ankle[1]-knee[1],2))

    shoulderRange = math.sqrt(pow(R_Shoulder[0]-L_Shoulder[0],2) + pow(R_Shoulder[1]-L_Shoulder[1],2))
    hipRange = math.sqrt(pow(R_hip[0]-L_hip[0],2) + pow(R_hip[1]-L_hip[1],2))
    reachA = math.sqrt(pow(L_Elbow[0]-L_Shoulder[0],2) + pow(L_Elbow[1]-L_Shoulder[1],2))
    reachB = math.sqrt(pow(L_Elbow[0]-L_Wrist[0],2) + pow(L_Elbow[1]-L_Wrist[1],2))
    reach = reachA + reachB

    legHeight = C+D


    height = A+B+C+D+nose2ear


    size_dict = {'shoulder':shoulderRange,
                 'hip':hipRange,
                 'reach':reach,
                 'nose2ear':nose2ear,
                 'height':height,
                 'leg':legHeight
                 }
    return img,size_dict




def getImageMesh(inputImg,posedata):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    virtual_crop_size = 500
    joint_set = 'coco'
    output_path = './result/'
    img_name = inputImg.split('/')[-1]  # '101570.jpg'
    img_path = inputImg
    cfg.DATASET.target_joint_set = joint_set
    cfg.MODEL.posenet_pretrained = False

    # prepare model
    mesh_model = SMPL()

    model, joint_regressor, joint_num, skeleton, graph_L, graph_perm_reverse = get_joint_setting(mesh_model, joint_category=joint_set)
    model = model.cuda()
    joint_regressor = torch.Tensor(joint_regressor).cuda()
    vis_skeleton = ((0, 1), (0, 2), (2, 4), (1, 3), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 17), (6, 17), (11, 18), (12, 18), (17, 18), (17, 0), (6, 8), (8, 10),)

    # prepare input image and 2d pose
    coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')

    orig_img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    orig_height, orig_width = orig_img.shape[:2]
    pose_vis_img = orig_img.copy()
    coco_joint_list = posedata

    # set depth order and colors of people
    c = coco_joint_list

    # manual nms for hhrnet 2d pose input, openpose input may not need this process
    min_diff = 20

    drawn_joints = []
    for idx in range(len(coco_joint_list)):
        # filtering
        pose_thr = 0.1
        coco_joint_img = np.asarray(coco_joint_list[idx])[:, :3]
        coco_joint_img = add_pelvis(coco_joint_img, coco_joints_name)
        coco_joint_img = add_neck(coco_joint_img, coco_joints_name)
        coco_joint_valid = (coco_joint_img[:, 2].copy().reshape(-1, 1) > pose_thr).astype(np.float32)
        # filter inaccurate inputs
        det_score = sum(coco_joint_img[:, 2])
        if det_score < 1.5:
            continue
        # filter filter the same targes
        tmp_joint_img = coco_joint_img.copy()
        continue_check = False
        for ddx in range(len(drawn_joints)):
            drawn_joint_img = drawn_joints[ddx]
            drawn_joint_val = (drawn_joint_img[:, 2].copy().reshape(-1, 1) > pose_thr).astype(np.float32)
            diff = np.abs(tmp_joint_img[:, :2] - drawn_joint_img[:, :2]) * coco_joint_valid * drawn_joint_val
            diff = diff[diff != 0]
            if diff.size == 0:
                continue_check = True
            elif diff.mean() < min_diff:
                continue_check = True
        if continue_check:
            continue
        drawn_joints.append(tmp_joint_img)

        # get camera parameters
        project_net = models.project_net.get_model(crop_size=virtual_crop_size).cuda()
        joint_input = coco_joint_img
        metadata = {'model':model,'graph_perm_reverse': graph_perm_reverse,'mesh_model':mesh_model,'joint_regressor':joint_regressor}
        out = optimize_cam_param(project_net, joint_input, crop_size=virtual_crop_size, metadata=metadata)


        # vis mesh
        color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
        mesh_img = render(out, orig_height, orig_width, orig_img, mesh_model.face, color)  # s[idx])
        cv2.imwrite(output_path + f'{img_name[:-4]}_mesh_{idx}.png', mesh_img)

        # vis 2d pose
        tmpkps = np.zeros((3, len(joint_input)))
        tmpkps[0, :], tmpkps[1, :], tmpkps[2, :] = joint_input[:, 0], joint_input[:, 1], 1
        # swap pevlis and thorax
        tmpkps[:, -1], tmpkps[:, -2] = tmpkps[:, -2].copy(), tmpkps[:, -1].copy()
        pose_vis_img = vis_coco_skeleton(pose_vis_img, tmpkps, vis_skeleton, color)  # s[idx])
        pose_vis_img,size_dict = getRangeJoint(pose_vis_img,tmpkps,out['bbox'])
        cv2.imwrite(output_path + f'{img_name[:-4]}_pose2d_{idx}.png', pose_vis_img)

        save_obj(out['mesh'], mesh_model.face, output_path + f'{img_name[:-4]}_mesh_{idx}.obj')
        return mesh_img,pose_vis_img,size_dict