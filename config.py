import argparse
import datetime
from easydict import EasyDict


def parser():
    parser = argparse.ArgumentParser()
    
    # dataset parameters
    parser.add_argument("--data_path", default="/data4/liangdong/detectron2/datasets/coco", type=str, help="Path to Coco dataset")
    parser.add_argument("--iou_threshold", type=float, default=0.4, help="set threshold to filter matched annotations with low IoU")
    # parser.add_argument("--class_ids", type=int, nargs='+', default=-1, help="select specified classes")
    parser.add_argument("--gt_num_points", type=int, default=2000, help="total sample size from each GT annotation")
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers load dataset')
    
    # training parameters
    parser.add_argument('--multi_gpu', nargs='+', type=int, default=[0], help='Use multiple gpus')
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--optimizer", default='adam', type=str, choices=['adam', 'sgd'], help="Training optimization strategy")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Training epoch")
    parser.add_argument("--number_curves", type=int, default=1, help="The number of sampling curves from detected binary mask")
    
    # network parameters
    parser.add_argument("--num_points", type=int, default='11', help="The number of points/vertexs on each curve")
    parser.add_argument("--feature_dim", type=int, default=128, help="The size of feature map inside the network")
    parser.add_argument("--dilated_conv", action="store_true", help="Using dilated convolution")
    parser.add_argument("--padding_mode", default='reflect', type=str, choices=['reflect', 'circular', 'replicate'], help="padding mode for 1D convolution")
    
    
    # directory and saving
    parser.add_argument('--reload_model_path', type=str, default='', help='optional reload model path')
    # parser.add_argument('--env', type=str, default="Seg_atlasnet", help='visdom environment')
    parser.add_argument('--visdom_port', type=int, default=9102, help="visdom port")
    
    config = parser.parse_args()
    
    date = str(datetime.datetime.now())
    config.date = date
    config.log_path = "log/{}".format(date)
    
    config = EasyDict(config.__dict__)
    return config
    
if __name__=="__main__":
    config = parser()
    print(config)