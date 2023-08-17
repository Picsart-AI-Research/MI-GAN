import json
import math
import os
import os.path as osp

import numpy as np
import numpy.random as npr
import pyspng
import torch
import torchvision.transforms as tvtrans
import PIL.Image
from PIL import Image, ImageDraw
from zipfile import ZipFile

from .common import *

PIL.Image.MAX_IMAGE_PIXELS = None


@regdataset()
class ffhq(ds_base):
    def init_load_info(self, cfg):
        self.root_dir = cfg.root_dir
        mode = cfg.mode
        allow_partial = cfg.allow_partial

        with open(osp.join(cfg.root_dir, 'ffhq-dataset-v2.json'), 'r') as f:
            jinfo = json.load(f)

        subset = mode.split('+')
        subset = [
            'training' if mi == 'train' else
            'validation' if mi == 'val' else
            None
            for mi in subset]

        self.load_info = []
        for _, ji in jinfo.items():
            if ji['category'] not in subset:
                continue
            
            impath = osp.join(cfg.root_dir, ji['image']['file_path'])
            if allow_partial:
                # We allow a partial dataset just for debugging
                if not osp.isfile(impath):
                    continue
            else:
                if not osp.isfile(impath):
                    raise ValueError

            if not (impath.endswith(".jpg") or impath.endswith(".png")):
                continue

            modetag = \
                '00_train' if ji['category'] == 'training' else \
                '50_val' if ji['category'] == 'validation' else \
                None
            filename = osp.basename(impath)
            filetag = osp.splitext(filename)[0]

            uid = '{}-{}'.format(modetag, filetag)
            info = {
                'unique_id': uid, 
                'filename': filename,
                'image_path': impath,
            }
            self.load_info.append(info)


@regloader()
class DefaultLoader(object):
    def __init__(self):
        super().__init__()

    @pre_loader_checkings('image')
    def __call__(self, path, element):
        data = PIL.Image.open(path).convert('RGB')
        data = tvtrans.ToTensor()(data)
        return data


@regloader()
class R512Loader(object):
    """
    Loader with resolution 512
    """
    def __init__(self):
        super().__init__()

    @pre_loader_checkings('image')
    def __call__(self, path, element):
        data = PIL.Image.open(path).convert('RGB')
        data = data.resize([512, 512], PIL.Image.BICUBIC)
        data = tvtrans.ToTensor()(data)
        return data

#############
# formatter #
#############


@regformat()
class DefaultFormatter(object):
    """
    This formatter is a direct replication of the original CoModGan TF code
    """
    def __init__(self):
        self.lod = 0 
        self.latent_dim = 512

    def __call__(self, element):
        x = element['image']
        x = (x-0.5)*2
        if self.lod != 0:
            c, h, w = x.shape
            y = x.view(c, h//2, 2, w//2, 2)        
            y = y.mean(dim=(2, 4), keepdim=True)
            y = y.repeat(1, 1, 2, 1, 2)
            y = y.view(c, h, w)
            x = x + (y-x)*self.lod
        latent = torch.randn([512])
        mask = RandomMask(512)[0]
        return x, latent, mask, element['unique_id']


@regformat()
class CenterMaskFormatter(object):
    """
    This formatter that use the same center mask
    """
    def __init__(self):
        self.latent_dim = 512

    def __call__(self, element):
        x = element['image']
        x = (x-0.5)*2
        _, h, w = x.shape
        latent = torch.randn([512])
        mask = np.ones([h, w]).astype(np.float32)
        mask[h//4:(h//4+h//2), w//4:(w//4+w//2)] = 0
        return x, latent, mask, element['unique_id']

########################
# RandomBrush for mask #
########################


def RandomBrush(
    max_tries,
    s,
    min_num_vertex=4,
    max_num_vertex=18,
    mean_angle=2*math.pi / 5,
    angle_range=2*math.pi / 15,
    min_width=12,
    max_width=48
):
    H, W = s, s
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask


def RandomMask(s, hole_range=[0,1]):
    coef = min(hole_range[0] + hole_range[1], 1.0)
    while True:
        mask = np.ones((s, s), np.uint8)

        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
            mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0

        def MultiFill(max_tries, max_size):
            for _ in range(np.random.randint(max_tries)):
                Fill(max_size)

        MultiFill(int(10 * coef), s // 2)
        MultiFill(int(5 * coef), s)
        mask = np.logical_and(mask, 1 - RandomBrush(int(20 * coef), s))
        hole_ratio = 1 - np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return mask[np.newaxis, ...].astype(np.float32)

###############
# ffhq_simple #
###############


@regdataset()
class ffhqsimple(ds_base):
    def init_load_info(self, cfg):
        self.root_dir = cfg.root_dir
        mode = cfg.mode

        if mode == 'train256':
            imagedir = 'ffhq256x256'
        else:
            raise ValueError

        self.load_info = []
        for subi in os.listdir(osp.join(self.root_dir, imagedir)):
            for fi in os.listdir(osp.join(self.root_dir, imagedir, subi)):
                if fi.find('.png') == -1:
                    continue
                uid = osp.splitext(fi)[0]
                info = {
                    'unique_id': uid, 
                    'filename' : fi,
                    'image_path' : osp.join(self.root_dir, imagedir, subi, fi)
                }
                self.load_info.append(info)


@regformat()
class ImageOnlyFormatter(object):
    def __init__(self, random_flip=False):
        self.random_flip = random_flip

    def __call__(self, element):
        x = (element['image']*2 - 1)
        if self.random_flip and npr.rand() < 0.5:
            x = x.flip(-1)
        return x, element['unique_id']

############
# ffhq_zip #
############


@regdataset()
class ffhqzip(ds_base):
    def init_load_info(self, cfg):
        self.root_dir = cfg.root_dir
        mode = cfg.mode
        self.mode = mode

        if mode in ['train256']:
            zipfile = 'ffhq256x256.zip'
            simple_split = [10000, 70000]
        elif mode in ['val256']:
            zipfile = 'ffhq256x256.zip'
            simple_split = [0, 10000]
        elif mode in ['train512', 'train512ori']:
            zipfile = 'ffhq512x512.zip'
            simple_split = [10000, 70000]
        elif mode in ['val512', 'val512ori']:
            zipfile = 'ffhq512x512.zip'
            simple_split = [0, 10000]
        else:
            raise ValueError

        self.load_info = []
        with ZipFile(osp.join(self.root_dir, zipfile), 'r') as z:
            for fi in z.namelist():
                if fi.find('.png') == -1:
                    continue
                filename = osp.basename(fi)
                uid = osp.splitext(filename)[0]
                info = {
                    'unique_id': uid, 
                    'filename' : filename,
                    'image_path' : fi,
                    'zipfile' : osp.join(self.root_dir, zipfile)
                }
                self.load_info.append(info)

        self.load_info = sorted(self.load_info, key=lambda x:x['unique_id'])
        if simple_split is not None:
            self.load_info = self.load_info[simple_split[0]:simple_split[1]]


@regloader()
class ZipLoader(object):
    def __init__(self):
        super().__init__()
        self.zipfile = None
        self.zipfilename = None

    @pre_loader_checkings('image')
    def __call__(self, path, element):
        if self.zipfilename != element['zipfile']:
            self.zipfile = ZipFile(element['zipfile'], 'r')
            self.zipfilename = element['zipfile']
        with self.zipfile.open(path, 'r') as f:
            data = pyspng.load(f.read())
        data = tvtrans.ToTensor()(data)
        return data

    def zipfile_close(self):
        if self.zipfile is not None:
            self.zipfile.close()
        self.zipfile = None
        self.zipfilename = None


@regformat()
class RandomMaskFormatter(object):
    """
    This formatter is a direct replication of the original CoModGan TF code
    """
    def __init__(self, random_flip=True, mask_resolution=256, hole_range=[0, 1]):
        self.random_flip = random_flip
        self.mask_resolution = mask_resolution
        self.hole_range = hole_range

    def __call__(self, element):
        x = (element['image'] * 2 - 1)
        if self.random_flip and npr.rand() < 0.5:
            x = x.flip(-1)
        mask = RandomMask(self.mask_resolution, self.hole_range)[0]
        return x, mask, element['unique_id']
