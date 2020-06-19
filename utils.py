import numpy as np
import cv2
from pycocotools.mask import frPyObjects, decode

from math import floor

def isometric_polygon_sample(polygons, interval=1):
        num_polygons = len(polygons)

        # each sampling number is proportional to its polygon perimeter
        poly_perimeter = []
        for polygon in polygons:
            num_edges, _ = polygon.shape
            mat = np.eye(num_edges)
            mat = -1 * mat + np.concatenate((mat[1:, :], mat[[0], :]), axis=0)
            perimeter = np.linalg.norm(mat.dot(polygon), axis=1).sum()
            poly_perimeter.append(perimeter)
            
        # get number sampling points
        number_points = floor(np.sum(poly_perimeter)//interval)
        poly_perimeter = poly_perimeter / sum(poly_perimeter)
        
        # main part of samples
        sample_nums_pri = np.floor(poly_perimeter * number_points).astype(int)
        # secondary part of samples
        nums_sec = number_points - np.sum(sample_nums_pri)
        num = nums_sec // num_polygons
        modulus = nums_sec % num_polygons

        sample_nums_sec = [num + 1] * modulus + [num] * (num_polygons - modulus)

        # sample from polygons
        sample_points = []
        for n1, n2, polygon in zip(sample_nums_pri, sample_nums_sec, polygons):
            n = n1 + n2
            if n == 0:
                continue

            num_edges, _ = polygon.shape

            # compute edge length
            mat = np.eye(num_edges)
            mat = -1 * mat + np.concatenate((mat[1:, :], mat[[0], :]), axis=0)

            dist = np.linalg.norm(mat.dot(polygon), axis=1)
            cum_dist = np.cumsum(dist)
            length = np.sum(dist)

            # uniform sampling from 0 to edges_length
            rd = length * np.random.rand(n)

            # r1: the integer part of rd, used as index
            r1 = np.sum(np.greater.outer(rd, cum_dist), axis=1)
            cum_dist_small = np.cumsum(np.concatenate([[0], dist[:-1]]))

            # r2: the demical part
            r2 = (rd - cum_dist_small[r1]) / dist[r1]

            # convert into edges
            points = polygon[r1, :]
            r2 = r2.reshape([n, 1])
            r2 = np.concatenate((r2, r2), axis=1)
            # points shape: [n, 2]
            displacement = mat.dot(polygon)[r1, :]
            points = points + np.multiply(r2, displacement)
            sample_points.append(points)

        return np.concatenate(sample_points, axis=0)


def fixed_polygon_sample(polygons, number_points):
        num_polygons = len(polygons)

        # each sampling number is proportional to its polygon perimeter
        poly_perimeter = []
        for polygon in polygons:
            num_edges, _ = polygon.shape
            mat = np.eye(num_edges)
            mat = -1 * mat + np.concatenate((mat[1:, :], mat[[0], :]), axis=0)
            perimeter = np.linalg.norm(mat.dot(polygon), axis=1).sum()
            poly_perimeter.append(perimeter)
            
        # polygons perimeter porpotion
        poly_perimeter = poly_perimeter / sum(poly_perimeter)
        
        # main part of samples
        sample_nums_pri = np.floor(poly_perimeter * number_points).astype(int)
        # secondary part of samples
        nums_sec = number_points - np.sum(sample_nums_pri)
        num = nums_sec // num_polygons
        modulus = nums_sec % num_polygons

        sample_nums_sec = [num + 1] * modulus + [num] * (num_polygons - modulus)

        # sample from polygons
        sample_points = []
        for n1, n2, polygon in zip(sample_nums_pri, sample_nums_sec, polygons):
            n = n1 + n2
            if n == 0:
                continue

            num_edges, _ = polygon.shape

            # compute edge length
            mat = np.eye(num_edges)
            mat = -1 * mat + np.concatenate((mat[1:, :], mat[[0], :]), axis=0)

            dist = np.linalg.norm(mat.dot(polygon), axis=1)
            cum_dist = np.cumsum(dist)
            length = np.sum(dist)

            # uniform sampling from 0 to edges_length
            rd = length * np.random.rand(n)

            # r1: the integer part of rd, used as index
            r1 = np.sum(np.greater.outer(rd, cum_dist), axis=1)
            cum_dist_small = np.cumsum(np.concatenate([[0], dist[:-1]]))

            # r2: the demical part
            r2 = (rd - cum_dist_small[r1]) / dist[r1]

            # convert into edges
            points = polygon[r1, :]
            r2 = r2.reshape([n, 1])
            r2 = np.concatenate((r2, r2), axis=1)
            # points shape: [n, 2]
            displacement = mat.dot(polygon)[r1, :]
            points = points + np.multiply(r2, displacement)
            sample_points.append(points)

        return np.concatenate(sample_points, axis=0)

def RLE_to_polygons(rle, approx="simple"):
    
    # convert RLE
    pred_mask = decode(rle)
    pred_mask = pred_mask.astype(np.uint8)
    
    # consider large mask with small scale
    
    pred_mask = cv2.copyMakeBorder(pred_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    # set approx method
    contour_approx = [cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_TC89_L1]
    if approx=="tc89":
        approx_method = contour_approx[2]
    elif approx=="simple":
        approx_method = contour_approx[1]
    else:
        approx_method = contour_approx[0]
    # set search method
    contour_search = [cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP, cv2.RETR_TREE]
    contour_search = cv2.RETR_LIST
    # apply findContours()
    polygons = cv2.findContours(pred_mask, contour_search, approx_method, offset=(-1, -1))
    polygons = polygons[0] if len(polygons) == 2 else polygons[1]
    return polygons


def mask_to_heatmap(mask):
    edge_map = np.abs(cv2.Laplacian(mask, cv2.CV_64F, ksize=1))
    # consider small mask with small kernal size
    heatmap = cv2.GaussianBlur(edge_map, (35,35), 6)
    return heatmap


def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    
    return (Ia.T*wa).T + (Ib.T*wb).T + (Ic.T*wc).T + (Id.T*wd).T


# given ksize get kernal
def polygon_conv(y, k_size):
    # kernal [1, ..., -1, ..., 1]
    k_half = k_size//2 + 1
    k = np.concatenate([np.linspace(1.0, -1, k_half)[:-1],np.linspace(-1, 1.0, k_half)], axis=0)

    y_ext = np.concatenate([y[-k_half+1:],y,y[:k_half-1]], axis=0)
    result = np.convolve(y_ext, np.exp(k), 'valid')

    # avoid negative result
    return np.abs(result)


def prob_curve_sample(polygons, heatmap, num_points):
    # store coordinates
    valid_polygons = []
    # store valid polygons size
    polygons_size = []
    # store probabilities
    prob_list = []
    for poly in polygons:
        poly = poly.reshape(-1, 2)
        poly_size = poly.shape[0]
        
        # filter small polygons
        if poly_size >= num_points:
            
            valid_polygons.append(poly)
            polygons_size.append(poly_size)
            # get heat and prob
            eps = 0.01
            heat_list = 1/(eps + bilinear_interpolate(heatmap, poly[:,0], poly[:,1]))
            prob_list.extend(polygon_conv(heat_list, num_points))

    # no valid polygons
    if not len(valid_polygons):
        return 
            
    # normalize probabilities
    prob_list = prob_list/np.sum(prob_list)
    
    # get sample indexes
    sampled_index = np.random.choice(np.arange(sum(polygons_size)), size=1, p=prob_list)
    
    poly_index = sum(sampled_index > np.cumsum(polygons_size))
    point_index = sampled_index - sum(polygons_size[:poly_index])
    
    # get curve
    num_half = num_points//2 
    sampled_polygon = valid_polygons[poly_index].copy()
    sampled_polygon = np.concatenate([sampled_polygon[-num_half:,:],
                                      sampled_polygon,
                                      sampled_polygon[:num_half,:]], axis=0)
    curve = sampled_polygon[int(point_index):int(point_index)+num_points,:]
    if curve.shape[0] < num_points:
        return
    return curve


