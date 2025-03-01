#-*-coding:utf-8-*-
# date:2024-08
# Author: Xian
# function: utils
import os
import numpy as np
import cv2
import copy

def resize_img_keep_ratio(img, target_size):

    old_size = img.shape[:2]  # 原始图像大小
    ratio = min(target_size[i] / old_size[i] for i in range(len(old_size)))  # 计算比例
    new_size = tuple(int(i * ratio) for i in old_size)  # 计算新的图像大小
    img = cv2.resize(img, (new_size[1], new_size[0]))  # 调整图像大小

    pad_w = target_size[1] - new_size[1]  # 计算宽度填充
    pad_h = target_size[0] - new_size[0]  # 计算高度填充
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(200,200,200))  # 边缘填充
    return img_new
'''
function：读取 obj 信息
'''
def read_obj(objFilePath):
    mesh = {
        "joints":None,
        "faces_index":None,
        }
    print("objFilePath:",objFilePath)
    with open(objFilePath) as file:
        joints = []
        faces_index = []
        while True:
            line = file.readline().strip()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v"and len(strs)==4:
                joints.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "f" and len(strs)==4:
                faces_index.append((int(strs[1]), int(strs[2]), int(strs[3])))

    print("joints num : {}".format(len(joints)))
    print("faces  num : {}".format(len(faces_index)))

    if len(joints)!=0:
        mesh["joints"] = np.array(joints).reshape(-1,3)

    if len(faces_index)!=0:
        mesh["faces_index"] = np.array(faces_index).reshape(-1,3)
    return mesh
