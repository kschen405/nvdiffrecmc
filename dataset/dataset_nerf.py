import os
import glob
import json

import torch
import numpy as np

from render import util

from dataset import Dataset

###############################################################################
# NERF image based dataset (synthetic)
###############################################################################

def _load_img(path):
    files = glob.glob(path + '.*')
    if len(files) == 0:
        files = glob.glob(path)
    # files = glob.glob(path + '.*')
    assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
    img = util.load_image_raw(files[0])
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

class DatasetNERF(Dataset):
    def __init__(self, cfg_path, cfg_studio_path, FLAGS, examples=None):
        print("Using class DatasetNERF")
        self.FLAGS = FLAGS
        self.examples = examples
        # Load config / transforms
        self.cfg = json.load(open(cfg_path, 'r'))
        self.cfg_studio = json.load(open(cfg_studio_path, 'r'))
        # assert FLAGS.nvdif_extrinsic
        if self.FLAGS.nvdif_extrinsic:
            print("Using nvdif's version dataset")
            self.base_dir = os.path.dirname(cfg_path)
            self.n_images = len(self.cfg['frames'])
            self.resolution = _load_img(os.path.join(self.base_dir, self.cfg['frames'][0]['file_path'])).shape[0:2]
            print("self.n_images = ", self.n_images)
        else:
            print("Using nerfstudio's version dataset")
            self.base_dir = os.path.dirname(cfg_studio_path)
            self.n_images = len(self.cfg_studio['frames'])
            self.resolution = _load_img(os.path.join(self.base_dir, self.cfg_studio['frames'][0]['file_path'])).shape[0:2]
            print("self.n_images = ", self.n_images)


        self.aspect = self.resolution[1] / self.resolution[0]
        # Compute relative transformation matrix
        # nerf_path = '/home/kc81/current_projects/nvdiffrecmc/data/nerf_synthetic/hotdog/transforms_train.json'
        # nerf_studio_path = '/home/kc81/current_projects/nvdiffrecmc/data/nerf_synthetic/hotdog_nerfstudio/train_out/transforms_train.json'
        # cfg = json.load(open(nerf_path, 'r'))
        # cfg_studio = json.load(open(nerf_studio_path, 'r'))
        # for i in [0, 1, 2]:
        i = 0
        file_path_to_find = f"./train/r_{i}"
        idx1 = -1
        for idx, frame in enumerate(self.cfg['frames']):
            if frame['file_path'] == file_path_to_find:
                idx1 = idx
                break
        P1 = self.cfg["frames"][idx1]["transform_matrix"]

        file_path_to_find = f"images/frame_{i+1:05d}.png"
        idx2 = -1
        for idx, frame in enumerate(self.cfg_studio['frames']):
            if frame['file_path'] == file_path_to_find:
                idx2 = idx
                break
        P2 = self.cfg_studio["frames"][idx2]["transform_matrix"]
        print('idx1 = ', idx1, 'idx2 = , ', idx2)
        print("P2P1-1 = ", np.matmul(P2, np.linalg.inv(P1)))
        print("P1P2-1 = ", np.matmul(P1, np.linalg.inv(P2)))
        self.rel_trans_matrix = np.matmul(P1, np.linalg.inv(P2))
        print('Our camera pose to nvdifmc pose:')
        print('self.rel_trans_matrix = ', self.rel_trans_matrix)
        ####################################################################################################################
        print("self.resolution = ", self.resolution)
        print("self.aspect = ", self.aspect)
        print("DatasetNERF: %d images with shape [%d, %d]" % (self.n_images, self.resolution[0], self.resolution[1]))

        # Pre-load from disc to avoid slow png parsing
        if self.FLAGS.pre_load:
            print("Preload images")
            self.preloaded_data = []
            for i in range(self.n_images):
                if self.FLAGS.nvdif_extrinsic:
                    self.preloaded_data += [self._parse_frame(self.cfg, i)]
                else:
                    self.preloaded_data += [self._parse_frame(self.cfg_studio, i)]

    def _parse_frame(self, cfg, idx):
        # Config projection matrix (static, so could be precomputed)
        if self.FLAGS.nvdif_intrinsic:
            print("Using nvdif's intrinsic")
            fovy   = util.fovx_to_fovy(self.cfg['camera_angle_x'], self.aspect)
            proj   = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        else:
            print("Using nerfstudio's intrinsic")
            camera_angle_x = 2 * np.arctan( self.cfg_studio['w'] / (2 * self.cfg_studio['fl_x']))
            fovy   = util.fovx_to_fovy(camera_angle_x, self.aspect)
            proj   = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Load image data and modelview matrix
        img    = _load_img(os.path.join(self.base_dir, cfg['frames'][idx]['file_path'])) # cfg can be either nvdif or studio
        if self.FLAGS.nvdif_extrinsic:
            mv     = torch.linalg.inv(torch.tensor(self.cfg['frames'][idx]['transform_matrix'], dtype=torch.float32))
        else: # mv: world to camera
            mv     = torch.linalg.inv(torch.tensor(np.matmul(self.rel_trans_matrix, self.cfg_studio['frames'][idx]['transform_matrix']), dtype=torch.float32))
            # mv     = torch.linalg.inv(torch.tensor(self.cfg_studio['frames'][idx]['transform_matrix'], dtype=torch.float32))

        mv     = mv @ util.rotate_x(-np.pi / 2)

        campos = torch.linalg.inv(mv)[:3, 3] # camera position in the world: x, y, z = translation x, y, z
        mvp    = proj @ mv # image <- camera <- world # move -> view -> project

        return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...] # Add batch dimension

    def getMesh(self):
        return None # There is no mesh

    def __len__(self):
        return self.n_images if self.examples is None else self.examples

    def __getitem__(self, itr):
        # img      = []
        # fovy     = util.fovx_to_fovy(self.cfg['camera_angle_x'], self.aspect)

        if self.FLAGS.pre_load:
            img, mv, mvp, campos = self.preloaded_data[itr % self.n_images]
        else:
            if self.FLAGS.nvdif_extrinsic: 
                img, mv, mvp, campos = self._parse_frame(self.cfg, itr % self.n_images)
            else:
                img, mv, mvp, campos = self._parse_frame(self.cfg_studio, itr % self.n_images)

        return {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : self.FLAGS.train_res,
            'spp' : self.FLAGS.spp,
            'img' : img
        }

# def _load_img(path):
#     files = glob.glob(path + '.*')
#     if len(files) == 0:
#         files = glob.glob(path)
#     assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
#     img = util.load_image_raw(files[0])
#     if img.dtype != np.float32: # LDR image
#         print('srgb_to_rgb')
#         img = torch.tensor(img / 255, dtype=torch.float32)
#         img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
#     else:
#         img = torch.tensor(img, dtype=torch.float32)
#     return img

# class DatasetNERF_Studio(Dataset):
#     def __init__(self, cfg_path, cfg_studio_path, FLAGS, examples=None):
#         print("Using class DatasetNERF_Studio")
#         self.FLAGS = FLAGS
#         self.examples = examples
#         assert not FLAGS.nvdif_extrinsic
#         if FLAGS.nvdif_extrinsic:
#             print("Using nvdif's version dataset")
#             self.base_dir = os.path.dirname(cfg_path)
#         else:
#             print("Using nerfstudio's version dataset")
#             self.base_dir = os.path.dirname(cfg_studio_path)

#         # Load config / transforms
#         self.cfg = json.load(open(cfg_path, 'r'))
#         self.cfg_studio = json.load(open(cfg_studio_path, 'r'))
#         self.n_images = len(self.cfg['frames'])

#         # Determine resolution & aspect ratio
#         self.resolution = _load_img(os.path.join(self.base_dir, self.cfg['frames'][0]['file_path'])).shape[0:2]
#         self.aspect = self.resolution[1] / self.resolution[0]

#         print("DatasetNERF: %d images with shape [%d, %d]" % (self.n_images, self.resolution[0], self.resolution[1]))

#         # Pre-load from disc to avoid slow png parsing
#         if self.FLAGS.pre_load:
#             self.preloaded_data = []
#             for i in range(self.n_images):
#                 self.preloaded_data += [self._parse_frame(self.cfg, i)]

#         # nvdif
#         w, h = int(self.cfg['w']), int(self.cfg['h'])
#         # self.img_wh = (w, h)
#         self.camera_angle_x = 2 * np.arctan( self.cfg['w'] / (2 * self.cfg['fl_x'])) # both value before downsample

#         # # Compute relative transformation matrix
#         # nerf_path = '/home/kc81/current_projects/nvdiffrecmc/data/nerf_synthetic/hotdog/transforms_train.json'
#         # nerf_studio_path = '/home/kc81/current_projects/nvdiffrecmc/data/nerf_synthetic/hotdog_nerfstudio/train_out/transforms_train.json'
#         # cfg = json.load(open(nerf_path, 'r'))
#         # cfg_studio = json.load(open(nerf_studio_path, 'r'))
#         # # for i in [0, 1, 2]:
#         # i = 0
#         # file_path_to_find = f"./train/r_{i}"
#         # idx1 = -1
#         # for idx, frame in enumerate(cfg['frames']):
#         #     if frame['file_path'] == file_path_to_find:
#         #         idx1 = idx
#         #         break
#         # P1 = cfg["frames"][idx1]["transform_matrix"]

#         # file_path_to_find = f"images/frame_{i+1:05d}.png"
#         # idx2 = -1
#         # for idx, frame in enumerate(cfg_studio['frames']):
#         #     if frame['file_path'] == file_path_to_find:
#         #         idx2 = idx
#         #         break
#         # P2 = cfg_studio["frames"][idx2]["transform_matrix"]
#         # print('idx1 = ', idx1, 'idx2 = , ', idx2)
#         # print("P2P1-1 = ", np.matmul(P2, np.linalg.inv(P1)))
#         # self.relative_transform_matrix = np.matmul(P1, np.linalg.inv(P2))
#         # print('Our camera pose to nvdifmc pose')
#         # self.aspect = h/w
#         # self.transform_matrix = np.array([self.cfg['frames'][i]['transform_matrix'] for i in range(len(frames))])
#         #

#         # # Visualize camera poses
#         # import vedo
#         # poses = []
#         # for i in range(3):
#         #     poses.append(self.cfg['frames'][i]['transform_matrix'])
#         # poses = poses.numpy()
#         # poses[:, :, -1] *= 10
#         # pos = poses[:, :, -1]
#         # arrow_len, s = 1, 1
#         # x_end   = pos + arrow_len * poses[:, :, 0]
#         # y_end   = pos + arrow_len * poses[:, :, 1]
#         # z_end   = pos + arrow_len * poses[:, :, 2]
        
#         # x = vedo.Arrows(pos, x_end, s=s, c='red')
#         # y = vedo.Arrows(pos, y_end, s=s, c='green')
#         # z = vedo.Arrows(pos, z_end, s=s, c='blue')
#         # print('Start vedo show')
#         # vedo.show(x,y,z, axes=1)
#         # print('End vedo show')

#     def _parse_frame(self, cfg, idx):
#         try:
#             fovy   = util.fovx_to_fovy(cfg['camera_angle_x'], self.aspect)
#             proj   = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
#         except:
#             fovy   = util.fovx_to_fovy(self.camera_angle_x, self.aspect)
#             proj   = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

#         # Load image data and modelview matrix
#         img    = _load_img(os.path.join(self.base_dir, cfg['frames'][idx]['file_path']))
#         mv     = torch.linalg.inv(torch.tensor(cfg['frames'][idx]['transform_matrix'], dtype=torch.float32))
#         mv     = mv @ util.rotate_x(-np.pi / 2)

#         campos = torch.linalg.inv(mv)[:3, 3]
#         mvp    = proj @ mv

#         return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...] # Add batch dimension

#     def getMesh(self):
#         return None # There is no mesh

#     def __len__(self):
#         return self.n_images if self.examples is None else self.examples

#     def __getitem__(self, itr):
#         img      = []
#         try:
#             fovy     = util.fovx_to_fovy(self.cfg['camera_angle_x'], self.aspect)
#         except:
#             fovy   = util.fovx_to_fovy(self.camera_angle_x, self.aspect)

#         if self.FLAGS.pre_load:
#             img, mv, mvp, campos = self.preloaded_data[itr % self.n_images]
#         else:
#             img, mv, mvp, campos = self._parse_frame(self.cfg, itr % self.n_images)

#         return {
#             'mv' : mv,
#             'mvp' : mvp,
#             'campos' : campos,
#             'resolution' : self.FLAGS.train_res,
#             'spp' : self.FLAGS.spp,
#             'img' : img
#         }
