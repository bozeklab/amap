import os, re
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import random
from PIL import Image
import tifffile
from utils import get_resolution, TARGET_RES


MAX_N_INST = 350

class InstDataset(Dataset):
    def __init__(self, npy_dir, train, validate, d=384, step=160, duplicate={}):
        self.npy_dir = npy_dir
        self.train = train
        self.validate = validate
        self.d = d
        self.int_count = 0

        self.step = step

        img_fls = list(filter(lambda x: re.match(r'(.+).npy', x), os.listdir(npy_dir)))
        self.img_prefs = []
        for i in range(len(img_fls)):
            m = re.match(r'(.+).npy', img_fls[i])
            pref = m.groups()[0]
            if pref in duplicate:
                self.img_prefs.append((pref, duplicate[pref]))
            else:
                self.img_prefs.append((pref, 1))
        print("%i %s images" % (len(self.img_prefs), ["test", "train"][train]))

        self.img_masks = []
        self.per_img = []
        for i, (img_pref, times) in enumerate(self.img_prefs):
            img_mask = np.load(os.path.join(npy_dir, img_pref + ".npy"))
            self.img_masks.append(img_mask)
            d_img = img_mask.shape[1]
            if train or validate:
                self.per_img.append(4 * times)
            else:
                steps_per_axis = (d_img - self.d) // self.step + 1
                self.per_img.append(steps_per_axis ** 2)
        self.per_img_cumsum = np.cumsum(self.per_img)
        logging.info('Creating dataset with %i files' % len(self.img_masks))


    def n_per_img(self, img_nb):
        return self.per_img[img_nb]


    def __len__(self):
        return np.sum(self.per_img)


    def n_imgs(self):
        return len(self.img_masks)

    def d_img(self, img_nb):
        return self.img_masks[img_nb].shape[1]

    def transform(self, img_mask):
        if self.train:
            i = random.randint(0,3)
            if i == 1:
                img_mask = np.flip(img_mask, axis=1)
            elif i == 2:
                img_mask = np.flip(img_mask, axis=2)
            elif i == 3:
                img_mask = np.flip(np.flip(img_mask, axis=1), axis=2)

            i = random.randint(0,3)
            img_mask = np.rot90(img_mask, k=i, axes=(1,2))

        img = img_mask[:1, :, :].copy()
        ins_mask = img_mask[1, :, :].copy()
        sem_mask = img_mask[2, :, :].copy()

        return img.astype(np.float32), sem_mask.astype(np.long), ins_mask.astype(np.float32)

    def _expand_ins_mask(self, mask):
        ins = np.unique(mask)
        ins = ins[ins > 0]
        res = np.zeros((MAX_N_INST, *mask.shape), dtype=np.float32)
        for i, ins_nb in enumerate(ins):
            m = np.zeros_like(mask)
            m[mask == ins_nb] = 1
            res[i,:] = m
        return res, len(ins)


    def __getitem__(self, i):
        self.int_count = (self.int_count + 1) % 10
        file_i = np.min(np.where(self.per_img_cumsum > i)[0])
        d_img = self.img_masks[file_i].shape[1]
        if self.train or self.validate:
            empty = True
            while empty:
                x, y = random.randint(0, d_img - self.d - 1), random.randint(0, d_img - self.d - 1)
                img_mask = self.img_masks[file_i][:,x:(x+self.d),y:(y+self.d)]
                empty = np.all(img_mask[1,:,:] == 0) and (self.int_count > 0) # every 10 can sample an empty window
        else:
            if file_i > 0:
                n = i - self.per_img_cumsum[file_i - 1]
            else:
                n = i
            steps_per_axis = (d_img - self.d) // self.step + 1
            x = self.step * (n // steps_per_axis)
            y = self.step * (n % steps_per_axis)
            img_mask = self.img_masks[file_i][:, x:(x + self.d), y:(y + self.d)]

        img, sem_mask, ins_mask = self.transform(img_mask)
        ins_mask, n_ins = self._expand_ins_mask(ins_mask)

        return {'image': torch.from_numpy(img), 'sem_mask': torch.from_numpy(sem_mask),
                'ins_mask': torch.from_numpy(ins_mask), 'n_ins': n_ins,
                'offs': torch.from_numpy(np.array([file_i, x,y], dtype=np.int))}


############# TEST DSET #####################################


class TestDataset(Dataset):
    def __init__(self, imgs_dir, d=384, step=128):
        self.imgs_dir = imgs_dir

        self.d = d
        self.int_count = 0
        self.step = step

        self.img_fls = []
        self.per_img = []
        for base, _, files in os.walk(imgs_dir):
            base = base[len(imgs_dir):]
            files = list(filter(lambda x: re.match(r'(.+).(tiff|tif)', x), files))
            for file in files:
                fn = os.path.join(base, file)
                self.img_fls.append(fn)
                sh = tifffile.imread(os.path.join(imgs_dir, fn)).shape
                res = get_resolution(os.path.join(imgs_dir, fn), sh[-1])
                scale = round(res / TARGET_RES, 2)
                sh = sh[-1] * scale
                steps_per_axis = int((sh - d) // self.step + 1)
                self.per_img.append(steps_per_axis ** 2)

        self.imgs = {}
        self.per_img_cumsum = np.cumsum(self.per_img)
        self.cur_img_i = -1
        print('Creating dataset with %i files' % len(self.img_fls), flush=True)


    def n_per_img(self, img_nb):
        return self.per_img[img_nb]


    def __len__(self):
        return np.sum(self.per_img)


    def n_imgs(self):
        return len(self.img_fls)

    def read_file(self, fn):
        img = tifffile.imread(os.path.join(self.imgs_dir, fn))
        if len(img.shape) > 3:
            img = np.max(img, axis=1)
        if (len(img.shape) == 3) and (img.shape[0] > 2):
            img = np.max(img, axis=0)
        if len(img.shape) > 2:
            img = img[0]
        img = Image.fromarray(img)
        w, h = img.size
        res = get_resolution(os.path.join(self.imgs_dir, fn), w)
        scale = round(res/TARGET_RES, 2)
        if scale != 1:
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small'
            img = img.resize((newW, newH))

        img = np.array(img)
        img = img.astype(np.float32)
        img = img / np.max(img)
        img = np.expand_dims(img.astype(np.float32), 0)
        return img

    def __getitem__(self, i):
        file_i = np.min(np.where(self.per_img_cumsum > i)[0])

        n = i
        if file_i > 0:
            n = i - self.per_img_cumsum[file_i - 1]

        fn = self.img_fls[file_i]
        if not fn in self.imgs:
            self.imgs[fn] = self.read_file(fn)

        d_img = self.imgs[fn].shape[1]
        steps_per_axis = (d_img - self.d) // self.step + 1
        x = self.step * (n // steps_per_axis)
        y = self.step * (n % steps_per_axis)

        img = self.imgs[fn][:, x:(x + self.d), y:(y + self.d)]

        return {'image': torch.from_numpy(img), 'offs': torch.from_numpy(np.array([file_i, x, y, d_img], dtype=np.int))}

