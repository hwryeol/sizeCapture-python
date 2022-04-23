import image2pose
import torch
import pose2mesh

import json

def getImageMesh(inputImgPath):
    joint_data = image2pose.getImageJoint(inputImgPath)
    zero_tensor = torch.zeros((len(joint_data)),17,2).to('cuda')
    posedata = torch.cat([joint_data,zero_tensor],dim=2)
    posedata = posedata.tolist()

    return pose2mesh.getImageMesh(inputImgPath, posedata)