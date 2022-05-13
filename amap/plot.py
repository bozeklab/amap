from PIL import Image
import numpy as np
import colorsys
import random


def fill_with_colors(im, m, ncomp, cols):
    for nc in range(1, ncomp+1):
        rgb = list(colorsys.hsv_to_rgb(cols[nc-1], 0.75, 1))
        rgb.append(0.5)
        xs, ys = np.where(m == nc)

        for i in range(xs.size):
            im[xs[i], ys[i], :] = rgb
    return im

def plot_ins_sem_labels(img, m, sem, ncomp, fl_out):
    im = np.zeros((*m.shape, 4))
    im_sem = np.zeros((*m.shape, 4))
    col_inds = [ float(nc)/ncomp for nc in range(ncomp) ]
    col_sem = [ float(nc)/3 for nc in range(3) ]
    random.shuffle(col_inds)

    im = fill_with_colors(im, m, ncomp, col_inds)
    im_sem = fill_with_colors(im_sem, sem, 3, col_sem)
    im = np.append(im, im_sem, axis=1)
    bg = np.append(img[0, :, :], img[0, :, :], axis=1)

    im = Image.fromarray((im*255).astype(np.uint8), mode="RGBA")
    bg = Image.fromarray(bg * 255).convert("RGBA")

    Image.alpha_composite(bg, im).save(fl_out)
