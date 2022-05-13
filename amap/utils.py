import numpy as np
import re, os
import tifffile


TARGET_RES = 0.022724609375

################ CHECKPOINT #########################

def find_last_checkpoint(dir_checkpoint):
    fls = list(filter(lambda x: re.match(r'cp_([0-9]+).pth', x), os.listdir(dir_checkpoint)))
    if len(fls) > 0:
        nbs = list(map(lambda x: int(re.match(r'cp_([0-9]+).pth', x).group(1)), fls))
        return np.max(nbs)
    else:
        return -1

################ TIF RESOLUTION #########################

def get_resolution(tif_fl, sh):
    tif = tifffile.TiffFile(tif_fl)
    if "XResolution" in tif.pages[0].tags:
        x1, x2 = tif.pages[0].tags["XResolution"].value
        if tif.pages[0].tags["ResolutionUnit"].value == 3: # RESUNIT.CENTIMETER
            x2 = x2 * 10000
        return (x2 / x1) * (tif.pages[0].shape[0] / sh)
    else:
        return TARGET_RES

################ TABLE OF TRAINING FILES WITH UNDERREPRESENTED FILES REPLICATED #########################

def make_duplicate_table(dirs):
    with open("../../duplicate.txt", 'w') as f:
        for dir, n in dirs:
            fls = list(filter(lambda x: re.match(r'(.+).npy', x), os.listdir(dir)))
            for fl in fls:
                pref = fl[:-4]
                f.write("%s;%i\n" % (pref, n))


def read_duplicate_table(fl):
    with open(fl, 'r') as f:
        lns = f.readlines()
    res = {}
    for ln in lns:
        t = ln.strip().split(";")
        res[t[0]] = int(t[1])
    return res

################ EXTRACT INST MASKS #########################

def merge_with_mask(mask_img, pred_mask, pred_sem, x, y):
    max_label = np.max(mask_img)
    pred_labels = np.unique(pred_mask)
    pred_labels = pred_labels[pred_labels > 0]
    mask_ins = mask_img[0,x:(x+pred_mask.shape[0]),y:(y+pred_mask.shape[1])]
    mask_sem = mask_img[1,x:(x+pred_mask.shape[0]),y:(y+pred_mask.shape[1])]
    for pred_l in pred_labels:
        is_pred_l = pred_mask == pred_l
        mask_values = mask_ins[is_pred_l]
        mask_ls, counts = np.unique(mask_values, return_counts=True)
        not_zero = mask_ls != 0
        enough_counts = (counts / np.sum(is_pred_l)) > 0.1
        mask_ls = mask_ls[not_zero & enough_counts]
        counts = counts[not_zero & enough_counts]
        if mask_ls.size == 0:
            mask_ins[is_pred_l] = max_label + 1
            max_label += 1
        else:
            mask_l = mask_ls[np.where(counts == np.max(counts))[0][0]]
            mask_ins[is_pred_l] = mask_l
    # merge semantic segm - maximum of class
    mask_sem = np.max(np.append(np.expand_dims(mask_sem, 0), np.expand_dims(pred_sem, 0), axis=0), axis=0)

    mask_img[0,x:(x + pred_mask.shape[0]), y:(y + pred_mask.shape[1])] = mask_ins
    mask_img[1,x:(x + pred_mask.shape[0]), y:(y + pred_mask.shape[1])] = mask_sem
    return mask_img

