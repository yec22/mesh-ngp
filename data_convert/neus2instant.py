
from genericpath import exists
import json
from os.path import join
import numpy as np
import os
import cv2
import sys
import torch
import torch.nn.functional as F
from glob import glob
import shutil
import argparse


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.conf = conf

        self.data_dir = conf['data_dir']
        self.render_cameras_name = conf['render_cameras_name']
        self.object_cameras_name = conf['object_cameras_name']

        self.camera_outside_sphere = True
        self.scale_mat_scale = 1.1

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis)

        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.intrinsics_all = torch.stack(self.intrinsics_all)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.fx = self.intrinsics_all[0][0, 0]
        self.fy = self.intrinsics_all[0][1, 1]
        self.cx = self.intrinsics_all[0][0, 2]
        self.cy = self.intrinsics_all[0][1, 2]
        print(self.fx, self.fy, self.cx, self.cy)
        self.pose_all = torch.stack(self.pose_all)  # [n_images, 4, 4]

        print('Load data: End')

def generate(dataset_name, base_par_dir, copy_image=True):
    base_dir = os.path.join(base_par_dir, dataset_name)
    
    output_dir = f'./data/neus/{dataset_name}'
    print(output_dir)

    conf = {
        "data_dir": base_dir,
        "render_cameras_name": "cameras_sphere.npz",
        "object_cameras_name": "cameras_sphere.npz",
    }
    dataset = Dataset(conf)
    image_name = 'image'
    mask_name = 'mask'

        
    os.makedirs(output_dir, exist_ok=True)

    base_rgb_dir = join(base_dir,image_name)
    base_msk_dir = join(base_dir, mask_name)
    all_images = sorted(os.listdir(base_rgb_dir))
    all_masks = sorted(os.listdir(base_msk_dir))
    assert len(all_images) == len(all_masks)
    print("#images:", len(all_images))

    H, W = 1200, 1600

    if copy_image:
        new_image_dir = join(output_dir, "images")
        os.makedirs(new_image_dir, exist_ok=True)
        for i in range(len(all_images)):
            img_name = all_images[i]
            msk_name = all_masks[i]
            img_path = join(base_rgb_dir, img_name)
            msk_path = join(base_msk_dir, msk_name)
            img = cv2.imread(img_path)
            msk = cv2.imread(msk_path, 0)
            image = np.concatenate([img,msk[:,:,np.newaxis]],axis=-1)
            H , W = image.shape[0], image.shape[1]
            H , W = image.shape[0], image.shape[1]
            cv2.imwrite(join(new_image_dir, img_name), image)
        print("Copy images done")
        base_rgb_dir = "images"

    print("base_rgb_dir:", base_rgb_dir)

    output = {
        "w": W,
        "h": H,
        "fl_x": dataset.fx.item(),
        "fl_y": dataset.fy.item(),
        "cx": dataset.cx.item(),
        "cy": dataset.cy.item(),
        "aabb_scale": 1.0,
    }

    # for test, we only use the 1st frame
    for frame_i in range(1):
        output['frames'] = []
        all_rgb_dir = sorted(os.listdir(join(output_dir,base_rgb_dir)))
        rgb_num = len(all_rgb_dir)
        camera_num = dataset.intrinsics_all.shape[0]
        assert rgb_num == camera_num, "The number of cameras should be eqaul to the number of images!"
        for i in range(rgb_num):
            rgb_dir = join(base_rgb_dir, all_rgb_dir[i])
            ixt = dataset.intrinsics_all[i]

            # add one_frame
            one_frame = {}
            one_frame["file_path"] = rgb_dir
            one_frame["transform_matrix"] = dataset.pose_all[i].tolist()

            output['frames'].append(one_frame)

        file_dir = join(output_dir,f'transforms_dtu.json')
        with open(file_dir,'w') as f:
            json.dump(output, f, indent=4)


if __name__ == "__main__":
    base_par_dir = "/data/yesheng/3D-scene/data/dtu"
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='dtu_scan105')
    args = parser.parse_args()
    
    generate(args.dataset_name, base_par_dir)