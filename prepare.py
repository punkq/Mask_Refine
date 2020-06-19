import numpy as np
import torch, torchvision
import cv2
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

from pycocotools.coco import COCO
from pycocotools import mask
import json


def get_mask_rcnn_model():
    # get mask rcnn model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "model_final_f10217.pkl"
    return DefaultPredictor(cfg)


def get_info(dataDir, dataType):
    # load coco imgs and annotations
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    coco = COCO(annFile)
    imgIds = coco.getImgIds()
    print("total image size:",len(imgIds))
    img_info_list = coco.loadImgs(imgIds)
    
    # get rcnn model
    predictor = get_mask_rcnn_model()
    
    # prepare info dict
    info = dict(dataDir=dataDir,dataType=dataType,trainingPairs=[])
    
    # load predictions and store training pairs
    with torch.no_grad():
        for img_info in img_info_list:
            # Prepare input image
            img = cv2.imread('{}/{}/{}'.format(dataDir,dataType,img_info['file_name']))
            #print("img shape:", img.shape)
            im_transformed = predictor.transform_gen.get_transform(img).apply_image(img)
            batched_inputs = [{"image": torch.as_tensor(im_transformed).permute(2, 0, 1)}]
            h, w = img_info['height'], img_info['width']
            
            # Get predictions 
            detected_instances = [x["instances"] for x in predictor.model.inference(batched_inputs)]
            # only consider one image, get the only prediction
            detected_instance = detected_instances[0]
            
            # take detected instance 1 by 1
            for i in range(len(detected_instance)):
                detected_thing_id = detected_instance.get("pred_classes")[i]
                detected_thing = coco_metadata.thing_classes[detected_thing_id]
                
                # get annotations
                catId = coco.getCatIds(detected_thing)
                annIds = coco.getAnnIds(imgIds=img_info["id"], catIds=catId)
                if len(annIds)>0:
                    # load predicted mask
                    pred_mask = detected_instance.get("pred_masks")[i,:,:].to("cpu").numpy()
                    pred_mask = np.array(pred_mask, dtype=np.uint8)
                    pred_mask = cv2.resize(pred_mask,dsize=(w,h),\
                                           interpolation=cv2.INTER_CUBIC)
                    pred_mask = np.asfortranarray(pred_mask) 
                    pred_rle = mask.encode(pred_mask)
                    
                    # prepare annotations
                    annos = coco.loadAnns(annIds)
                    ann_rle = []
                    for ann in annos:
                        rle = mask.frPyObjects(ann['segmentation'], h, w)
                        if type(rle) == list:
                            ann_rle.append(mask.merge(rle, intersect=False))
                        else:
                            ann_rle.append(rle)
                    
                    # compute iou and match annotations
                    IoU = mask.iou(ann_rle, [pred_rle], [ann['iscrowd'] for ann in annos])
                    best_IoU = np.max(IoU)
                    matched_ann_id = annIds[np.argmax(IoU)]
                    
                    # convert bytes to utf-8 string
                    pred_rle['counts'] = pred_rle['counts'].decode('utf8')
                
                    # store info
                    info["trainingPairs"].append(dict(image_id=img_info['id'],\
                                                    pred_class_id=int(detected_thing_id),\
                                                    pred_mask_rle=pred_rle,\
                                                    annotation_id=matched_ann_id,\
                                                    ann_iou=best_IoU))
    return info


if __name__=="__main__":
    # generate Mask R-CNN results
    # match detected instance and GT instance
    
    # path setting
    dataDir='/data4/liangdong/detectron2/datasets/coco'
    dataType='train2017'
    
    info = get_info(dataDir, dataType)
    # save info
    with open("prepare_{}_test.json".format(dataType), 'w') as f:
        f.write(json.dumps(info, indent=4))