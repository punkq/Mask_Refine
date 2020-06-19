import json
import numpy as np
import cv2
from scipy.interpolate import interp2d
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.mask import frPyObjects, decode
from imantics import Mask
import datetime
import torch


from utils import *
from config import parser

class AnnPolygon(Dataset):
    def __init__(self, config, train=True):
        super().__init__()
         
        # set dataset file
        self.dataDir = config.data_path
        if not train:
            self.dataType = 'val2017'
        else:
            self.dataType = 'train2017'
        self.ann_file = "prepare_{}_test.json".format(self.dataType)
        self.coco = COCO("{}/annotations/instances_{}.json".format(self.dataDir, self.dataType))
        self.num_points = config.num_points
        
        self.gt_num_points = config.gt_num_points
        
        # read annotation file
        self.ann = []
        with open(self.ann_file, 'r') as f:
            print("Reading Annotaion File Success!")
            anns_list = json.loads(f.read())["trainingPairs"]
        
        # filter small IoU or small annos
        self.ann = [ann for ann in anns_list if ann["ann_iou"] > config.iou_threshold]
        self.anno_size = len(self.ann)
        print("Successfully generate dataset! \nThere are", self.anno_size, "annotations.")
        
        # store invalid index
        self.invalid_ind = []

        
    def __len__(self):
        return self.anno_size
        
    def __getitem__(self, index):
        # show processing schedule
        # if index in np.floor(np.linspace(0, self.anno_size)):
        #     print("Dataloader is processing {} over {} annotations ...".format(index, self.anno_size))
        
        # Example of trainingPairs:
        """
            "image_id": 397133,
            "pred_class_id": 44,
            "pred_mask_rle": {
                "size": [
                    427,
                    640
                ],
                "counts": "bnh14W=1O2O10O00010O00100O100O001Mgd[6"
            },
            "annotation_id": 2105658,
            "ann_iou": 0.0
        """
        # time1 = datetime.datetime.now()
        
        annotation = self.ann[index]
        height, width = annotation["pred_mask_rle"]["size"]
        # given annotation id sample point cloud
        coco_ann = self.coco.loadAnns(annotation["annotation_id"])[0]

        if coco_ann['iscrowd']:
            # RLE
            RLE = frPyObjects(coco_ann['segmentation'], height, width)
            GT_mask = decode(RLE)
            GT_polygons = Mask(GT_mask).polygons()
        else:
            # polygon
            GT_polygons = coco_ann['segmentation']
            RLE = frPyObjects(coco_ann['segmentation'], height, width)
            GT_mask = decode(RLE)
            
        # multiple polygons result in GT_mask.shape:[H, W, N] where N>=2
        if len(GT_mask.shape) > 2:
            GT_mask = np.sum(GT_mask, axis=2)
        GT_mask = GT_mask.reshape(GT_mask.shape + (1,))
            
        # get GT polygons and isometric sampled points (point cloud)
        GT_polygons = [np.array(poly).reshape(-1,2) for poly in GT_polygons]
        GT_points = fixed_polygon_sample(GT_polygons, self.gt_num_points)

        # given rle sample one curve
        pred_polygons = RLE_to_polygons(annotation["pred_mask_rle"])
        GT_mask = GT_mask.astype(np.uint8)
        GT_heatmap = mask_to_heatmap(GT_mask)
        
        # sample based on heatmap, only sample one curve, (C_dim = 2)
        curve_coord = prob_curve_sample(pred_polygons, GT_heatmap, self.num_points)
        
        # if sample failed, the reason could be insufficient points of polygons 
        # record invalid annotaion id, try resample from other availble annotation
        if curve_coord is None:
            self.invalid_ind.append(index)
            while True:
                new_index = np.random.randint(0, self.anno_size)
                if new_index not in self.invalid_ind:
                    break
            return self.__getitem__(new_index)
        
        # time2 = datetime.datetime.now()
        
        # get features from RCNN mask (C_dim = 1)
        mask_features = bilinear_interpolate(GT_mask, curve_coord[:,0], curve_coord[:,1])
        img_file = "{}/{}/{}.jpg".format(self.dataDir, self.dataType, str(annotation["image_id"]).zfill(12))

        # get features from image, cv2 imread "BGR" (C_dim = 3)
        img = cv2.imread(img_file)
        image_features = bilinear_interpolate(img, curve_coord[:,0], curve_coord[:,1])

        # aggregate features (C_dim = 2 + 1 + 3)
        curve = np.concatenate([curve_coord, mask_features, image_features], axis=1)
        
        #dict(curve=curve, GT=GT_points, pred_mask_rle=annotation["pred_mask_rle"], img=img)
        # time3 = datetime.datetime.now()
        
        # print("curve shape:", curve.shape)
        # print("GT points shape:", GT_points.shape)
        return curve, GT_points
    
    def test_time(self, num=100):
        import random
        import pandas
        random.shuffle(self.ann)
        
        time_data = []
        for i in range(num):
            curve, GT_points = self.__getitem__(i)
            # curve, GT_points, time_dict = self.__getitem__(i)
        #     time_data.append(time_dict)
        # time_data = pandas.DataFrame(time_data)
        # print(time_data.describe())
        
    
    def show_invalid(self):
        print("invalid index size:", len(self.invalid_ind))
        print( self.invalid_ind)

if __name__=="__main__":
    config = parser()
    dataset = AnnPolygon(config, train=False)
    dataset.test_time()

    