import numpy as np
import re, os
import math
import pandas as pd
from utils import get_resolution
import argparse
import cv2
from skimage.morphology import skeletonize

########## FOOT PROCESS PARAMETERS ################

def fp_parameters(region, res):
    region = region.astype(np.uint8)
    _, contours, hier = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    area = cv2.contourArea(cnt) * res**2
    per = cv2.arcLength(cnt,True) * res
    if per == 0:
        return -1, -1, -1
    circ = (4 * math.pi * area) / (per**2)
    if circ > 1:
        print(circ)
    return per, area, circ


def fp_param_table(img_dr="../samples", pred_dr="../amap_res", out_dr="../amap_res/morphometry"):
    fls = list(filter(lambda x: re.match(r'(.+)_pred.npy', x), os.listdir(pred_dr)))
    prefs = [re.match(r'(.+)_pred.npy', x).group(1) for x in fls]
    prefs.sort()
    if not os.path.exists(out_dr):
        os.mkdir(out_dr)
    for i, pref in enumerate(prefs):
        print("%i / %i %s" % (i, len(prefs), pref))
        preds = np.load(os.path.join(pred_dr, pref + "_pred.npy" ))

        ins_pred = preds[0,:,:]
        values = np.unique(ins_pred)
        values = values[values != 0]
        resolution = get_resolution(os.path.join(img_dr, pref+".tif"), preds.shape[1])
        with open(os.path.join(out_dr, pref + "_fp_params.xls"), 'w') as f:
            f.write("Label\tArea\tPerim.\tCirc.\tFeret\n")
            for value in values:
                is_value = ins_pred == value
                per, area, circ = fp_parameters(is_value, resolution)
                f.write("%i\t%.3f\t%.3f\t%.3f\n" % (value, area, per, circ))
            f.close()


########## ROI ESTIMATION ################

def get_ROI_from_pred(preds, img_sh):
    MIN_AREA = 500
    preds = cv2.resize(preds, img_sh, interpolation=cv2.INTER_NEAREST)
    all_pred = preds > 0
    all_pred = all_pred.astype(np.uint8)

    _, contours, _ = cv2.findContours(all_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conv_cnt = [ cv2.convexHull(cont) for cont in contours ]
    # filter small areas
    contours = list(filter(lambda cnt: cv2.contourArea(cnt) > MIN_AREA, conv_cnt))

    mask_roi = np.zeros(img_sh, np.uint8)
    for i in range(len(contours)):
        mask1 = cv2.drawContours(np.zeros(img_sh, np.uint8), contours, i, 1, -1)
        mask_roi[mask1 == 1] = 1

    mask_orig = mask_roi.copy()
    kernel = np.zeros((11, 11), np.uint8)
    kernel = cv2.circle(kernel, (5,5), 5, 1, 0)

    mask_roi = cv2.dilate(mask_roi, kernel, iterations=15)
    mask_roi = cv2.erode(mask_roi, kernel, iterations=10)
    mask_roi[mask_orig == 1] = 1

    sd = preds.copy()
    sd[sd == 1] = 0
    sd[sd == 2] = 1
    sd[mask_roi == 0] = 0
    sd = skeletonize(sd)
    sd = sd.astype(np.uint8)

    return mask_roi, sd


#################### SKELETON LENGTH ###########################

END_POINT = 2
JUNCTION_POINT = 3
SLAB_POINT = 4

def skeleton_length(input_image, res):
    tagged_image = tag_image(input_image)
    return mark_trees(tagged_image, res)


def get_number_of_neighbors(image, x, y):
    return np.sum(image[max(0,x-1):min(image.shape[0],x+2), max(0,y-1):min(image.shape[1],y+2)]) - image[x,y]


def tag_image(input_image):
    output_image = np.zeros_like(input_image)

    for x in range(input_image.shape[0]):
        for y in range(0, input_image.shape[1]):
            if input_image[x,y] > 0:
                num_neighbors = get_number_of_neighbors(input_image, x, y)
                if (num_neighbors < 2):
                    output_image[x, y] = END_POINT
                elif (num_neighbors > 2):
                    output_image[x, y] = JUNCTION_POINT
                else:
                    output_image[x, y] = SLAB_POINT
    return output_image



def mark_trees(tagged_image, res):
    colored_image = np.zeros_like(tagged_image, dtype=int)
    visited_image = np.zeros_like(tagged_image)
    color = 0
    distances = []

    end_point_xs, end_point_ys  = np.where(tagged_image == END_POINT)
    # Visit trees starting at end points
    for i in range(end_point_xs.size):
        x, y = end_point_xs[i], end_point_ys[i]
        if visited_image[x,y] == 0:
            colored_image, visited_image, dist = visit_tree(x, y, tagged_image, colored_image, visited_image, color, res)
            distances.append(dist)
            color += 1

    jun_point_xs, jun_point_ys = np.where(tagged_image == JUNCTION_POINT)
    for i in range(jun_point_xs.size):
        x, y = jun_point_xs[i], jun_point_ys[i]
        if visited_image[x,y] == 0:
            colored_image, visited_image, dist = visit_tree(x, y, tagged_image, colored_image, visited_image, color, res)
            distances.append(dist)
            color += 1

    # Check for unvisited slab voxels in case there are circular trees without junctions
    slab_point_xs, slab_point_ys = np.where(tagged_image == SLAB_POINT)
    for i in range(slab_point_xs.size):
        x, y = slab_point_xs[i], slab_point_ys[i]
        if visited_image[x,y] == 0:
            # Mark that voxel as the start point of the circularskeleton
            colored_image, visited_image, dist = visit_tree(x, y, tagged_image, colored_image, visited_image, color, res)
            distances.append(dist)
            if np.any(colored_image == color):
                color += 1

    return colored_image, distances


def find_unvisited(x, y, tagged_image, visited_image):
    for i in range(max(0, x-1),min(visited_image.shape[0], x+2)):
        for j in range(max(0, y-1), min(visited_image.shape[1], y+2)):
            if ((i != x) or (j != y)) and (visited_image[i,j] == 0) and (tagged_image[i,j] > 0):
                return i, j
    return -1,-1


def distance(x1, y1, x2, y2, res):
    dx = (x1 - x2) * res
    dy = (y1 - y2) * res
    return (dx**2 + dy**2)**0.5


def visit_tree(x, y, tagged_image, colored_image, visited_image, color, res):
    colored_image[x,y] = color
    dist = 0
    to_revisit = []
    if tagged_image[x, y] == JUNCTION_POINT:
        to_revisit.append((x,y))

    next_x, next_y = find_unvisited(x,y, tagged_image, visited_image)
    prev_x, prev_y = x, y
    visited_image[prev_x, prev_y] = 1
    while (next_x >= 0) or (len(to_revisit) > 0):
        if next_x >= 0:
            visited_image[next_x, next_y] = 1
            colored_image[next_x, next_y] = color
            dist += distance(prev_x, prev_y, next_x, next_y, res)
            if tagged_image[next_x, next_y] == JUNCTION_POINT:
                to_revisit.append((next_x, next_y))
            prev_x, prev_y = next_x, next_y
            next_x, next_y = find_unvisited(next_x, next_y, tagged_image, visited_image)
        else:
            prev_x, prev_y = to_revisit[0]
            next_x, next_y = find_unvisited(prev_x, prev_y, tagged_image, visited_image)
            if next_x < 0:
                to_revisit.remove((prev_x, prev_y))
    return colored_image, visited_image, dist


############################# CALCULATE SD PARAMETERS #######################################

def take_middle_points(pts):
    res = [pts[0]]
    ds = pts[1:] - pts[:-1]
    last_non0 = 0
    for i in range(1, ds.size):
        if ds[i] != 1:
            #print(pts[(last_non0+1):(i+1)])
            res.append(np.mean(pts[(last_non0+1):(i+1)]))
            last_non0 = i
    res.append(np.mean(pts[(last_non0 + 1):(pts.size)]))
    return np.array(res)


def calculate_grid(sd, res):
    grid_d = 0.75 / res
    grid_steps = np.round(np.arange(0, sd.shape[0], grid_d)).astype(int)
    grid_steps = grid_steps[grid_steps < sd.shape[0]]
    all_ds = np.zeros((0))
    all_pts = 0
    for step in grid_steps:
        pts = np.where(sd[step,:] == 1)[0]
        if pts.size > 1:
            pts = take_middle_points(pts)
            all_pts += pts.size
            ds = pts[1:] - pts[:-1]
            ds = ds * res
            all_ds = np.append(all_ds, ds)
        pts = np.where(sd[:,step] == 1)[0]
        if pts.size > 1:
            pts = take_middle_points(pts)
            all_pts += pts.size
            ds = pts[1:] - pts[:-1]
            ds = ds * res
            all_ds = np.append(all_ds, ds)
    return all_pts, np.mean(all_ds)


def sd_length_grid_index(pred_dr="../amap_res", img_dr="../samples", out_dr="../amap_res/morphometry"):
    fls = list(filter(lambda x: re.match(r'(.+)_pred.npy', x), os.listdir(pred_dr)))
    prefs = [re.match(r'(.+)_pred.npy', x).group(1) for x in fls]
    f = open(os.path.join(out_dr, "SD_length_grid_index.xls"), 'w')
    f.write("%s\t%s\t%s\t%s\t%s\t%s\n" % ("file", "SD length", "grid crossings", "mean distance", "SD total length", "ROI total area"))
    for i, pref in enumerate(prefs):
        preds = np.load(os.path.join(pred_dr, pref + "_pred.npy"))
        sd = preds[1, :, :]
        roi_mask, sd = get_ROI_from_pred(preds[1,:,:], sd.shape)

        _, contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = 0
        for cnt in contours:
            roi_area += cv2.contourArea(cnt)

        res = get_resolution(os.path.join(img_dr, pref+".tif"), sd.shape[0])
        _, distances = skeleton_length(sd, res)
        total_sd_len = np.sum(distances)

        total_roi_area = roi_area * res ** 2

        sd_len = total_sd_len / total_roi_area
        grid_points, grid_index = calculate_grid(sd, res)

        f.write("%s\t%.3f\t%i\t%.3f\t%.3f\t%.3f\n" % (pref, sd_len, grid_points, grid_index, total_sd_len, total_roi_area))
    f.close()

############### COMBINE FP and SD PARAMETERS #####################

def combine_FP_SD(param_dr="../amap_res/morphometry"):
    t = pd.read_table(os.path.join(param_dr, "SD_length_grid_index.xls"))
    fp_area = np.zeros((t.shape[0]))
    fp_perim = np.zeros((t.shape[0]))
    fp_circ = np.zeros((t.shape[0]))
    for i in range(t.shape[0]):
        fl = t["file"][i]
        fp_t = np.loadtxt(os.path.join(param_dr, fl+"_fp_params.xls"), delimiter="\t", skiprows=1, ndmin=2)
        if fp_t.size > 0:
            fp_area[i] = np.mean(fp_t[:,1])
            fp_perim[i] = np.mean(fp_t[:,2])
            fp_circ[i] = np.mean(fp_t[:,3])
        else:
            fp_area[i] = 0
            fp_perim[i] = 0
            fp_circ[i] = 0
    t["FP Area"] = fp_area
    t["FP Perim."] = fp_perim
    t["FP Circ."] = fp_circ
    t.to_csv(os.path.join(param_dr, "all_params.xls"), sep="\t")


###### RUN PARAMETER ESTIMATION ############

def get_args():
    parser = argparse.ArgumentParser(description='Calculate morphometric parameters ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--img_dr', dest='img_dr', type=str, default="../sample_images",
                        help='Original image folder')
    parser.add_argument('-p', '--pred_dr', dest='pred_dr', type=str, default="../amap_res",
                        help='AMAP segmentation results folder')
    parser.add_argument('-o', '--out_dr', dest='out_dr', type=str, default="../amap_res/morphometry",
                        help='Output folder for morphometric parameters')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    print("Estimating foot process parameters...")
    fp_param_table(args.img_dr, args.pred_dr, args.out_dr)
    print("Estimating slit diaphragm parameters...")
    sd_length_grid_index(args.pred_dr, args.img_dr, args.out_dr)
    print("Combining parameters...")
    combine_FP_SD(args.out_dr)










