import numpy as np
import re, os
from plot import plot_ins_sem_labels
from scipy.ndimage.measurements import label
import cv2
from skimage.morphology import skeletonize
import tifffile
from utils import get_resolution, TARGET_RES
import argparse

def thinner_SD(m_sd, m_fp):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # skeletonize SD and dilate by 3 pixels
    skeleton = (skeletonize(m_sd)).astype(np.uint8)
    m_sd = cv2.dilate(skeleton, kernel, iterations=1)
    m_sd[m_sd > 0] = 1

    m_fp = cv2.dilate(m_fp, kernel, iterations=2)
    m_fp[m_sd > 0] = 0
    m_fp[m_fp > 0] = 1
    return m_sd, m_fp


def make_train_files(img_dr, labels_dr, out_dr, nephrin_channel, min_pixels, tmp_dir):
    fls = list(filter(lambda x: re.match(r'(.+).tif', x), os.listdir(img_dr)))
    prefs = list(map(lambda x: re.match(r'(.+).tif', x).group(1), fls))
    print("Found %i labeled image files" % len(prefs))
    if not os.path.exists(out_dr):
        os.mkdir(out_dr)
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    for pref in prefs:
        print("Converting %s ..." % pref)
        fl = os.path.join(img_dr, pref + ".tif")
        img = tifffile.imread(fl)
        if len(img.shape) > 2:
            img = img[nephrin_channel]

        m_sem = cv2.imread(os.path.join(labels_dr, pref + ".tif"), cv2.IMREAD_GRAYSCALE)
        m_sem[m_sem < 45] = 0
        m_sem[m_sem > 160] = 2
        m_sem[m_sem >= 45] = 1

        res = get_resolution(fl, img.shape[-1])
        scale = round(res / TARGET_RES, 2)

        m_sd = np.zeros_like(m_sem)
        m_sd[m_sem == 2] = 1
        m_fp = np.zeros_like(m_sem)
        m_fp[m_sem == 1] = 1

        if scale != 1:
            new_w, new_h = int(scale * img.shape[1]), int(scale * img.shape[0])
            assert new_w > 0 and new_h > 0, 'Scale is too small'
            img = cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)
            m_sd = cv2.resize(m_sd, (new_w, new_h), cv2.INTER_NEAREST)
            m_fp = cv2.resize(m_fp, (new_w, new_h), cv2.INTER_NEAREST)

        m_sd, m_fp = thinner_SD(m_sd, m_fp)
        m_fp[m_sd > 0] = 0
        m_sem = m_sd * 2 + m_fp
        img = img.astype(np.float32)
        img = img / np.max(img)

        m = m_sem.copy()
        m[m == 2] = 0

        m, ncomponents = label(m, np.ones((3, 3), dtype=int))
        nbs, counts = np.unique(m, return_counts=True)
        rem = False
        for i in range(nbs.size):
            if counts[i] < min_pixels:
                nb = nbs[i]
                nbs[i] = -1
                m_sem[m == nb] = 0
                m[m == nb] = 0
                rem = True
        if rem:
            m_old = m.copy()
            nb = 1
            for nb_old in nbs:
                if nb_old > 0:
                    m[m_old == nb_old] = nb
                    nb += 1

        img_mask = np.concatenate([np.expand_dims(img, axis=0), np.expand_dims(m, axis=0), np.expand_dims(m_sem, axis=0)])
        np.save(os.path.join(out_dr, "%s.npy" % (pref)), img_mask)
        plot_ins_sem_labels(img_mask[:1,:,:], img_mask[1,:,:], img_mask[2,:,:], ncomponents, os.path.join(tmp_dir, "%s_masks.png" % (pref)))

def get_args():
    parser = argparse.ArgumentParser(description='Generate binary files for the training dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--img_dr', dest='img_dr', type=str, default="../sample_images",
                        help='Folder with raw images in tif format')
    parser.add_argument('-l', '--labels_dr', dest='labels_dr', type=str, default="../sample_images/labels",
                        help='Folder with label segmentation masks of the raw images. Filenames should match.')
    parser.add_argument('-o', '--out_dr', dest='out_dr', type=str, default="../data",
                        help='Output folder.')
    parser.add_argument('-t', '--tmp_dr', dest='tmp_dr', type=str, default="../tmp",
                        help='Folder for images of the generated labels.')
    parser.add_argument('-nc', '--nephrin_channel', type=int, default=0,
                        help='Number of the channel to use for segmentation.', dest='nephrin_channel')
    parser.add_argument('-mp', '--min_pixels', type=int, default=10,
                        help='Smallest possible size of a foot process in pixels. Pixel blobs that are smaller will be removed as potential noise.', dest='min_pixels')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    make_train_files(args.img_dr, args.labels_dr, args.out_dr, args.nephrin_channel, args.min_pixels, args.tmp_dr)

