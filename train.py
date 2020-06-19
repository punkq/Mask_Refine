import ChamferDistancePytorch.chamfer2D.dist_chamfer_2D as dist_chamfer_2D
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from torchnet import meter
from tqdm import tqdm

from dataset import AnnPolygon
from config import parser
from network import Snake
from vis import Visualizer

class Trainer():
    def __init__(self, config):
        
        # data
        self.train_data = AnnPolygon(config, train=True)
        self.val_data = AnnPolygon(config, train=False)
        self.num_points = config.num_points
        self.gt_num_points = config.gt_num_points
        
        # model
        self.model = Snake(state_dim=config.batch_size, feature_dim=6, conv_type='dgrid')
            
        # Run on GPU/CPU
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{config.multi_gpu[0]}")
        else:
            self.device = torch.device(f"cpu")
            
        # check if load from existing model
        if config.reload_model_path:
            self.model.load(config.reload_model_path)
        else:
            self.model.cuda()
            
        self.train_dataloader = DataLoader(self.train_data, config.batch_size, shuffle=True,
                                           num_workers=config.num_workers, drop_last=True)
        self.val_dataloader = DataLoader(self.val_data, config.batch_size, shuffle=True,
                                           num_workers=config.num_workers, drop_last=True)

        # optimizer
        self.criterion = dist_chamfer_2D.chamfer_2DDist()
        if config.optimizer == "adam":
            self.lr = config.learning_rate
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        elif config.optimizer == "sgd":
            self.lr = config.learning_rate / 10
            self.optimizer = SGD(self.model.parameters(), lr=self.lr)
            
        self.train_loss = meter.AverageValueMeter()
        self.val_loss = meter.AverageValueMeter()
        
        self.epochs = config.epochs
        
        self.vis = Visualizer(config)
    
    
    def train(self):
        for epoch in range(self.epochs):
            self.train_loss.reset()
            # training iteration
            for i, (curve, GT) in enumerate(self.train_dataloader):
                # load
                GT_points = GT.type(torch.FloatTensor).cuda(device=self.device)
                curve = curve.type(torch.FloatTensor).cuda(device=self.device)
                coodinates = curve[:,:,:2]
                
                curve = curve.permute(0, 2, 1)
                coodinates = coodinates.permute(0, 2, 1)
                
                # feed into model
                offset = self.model(curve)
                new_coodinates = offset + coodinates
                
                dist1, dist2, _, _ = self.criterion(GT_points, new_coodinates.permute(0,2,1))
                loss = torch.mean(dist1)/self.num_points + torch.mean(dist2)/self.gt_num_points
                
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.train_loss.add(loss.item())
                self.vis.plot('train_loss', self.train_loss.value()[0], i+epoch*len(self.train_data))
                
            self.model.eval()
            
            for i, (curve, GT)  in enumerate(self.val_dataloader):
                # load data
                GT_points = GT.type(torch.FloatTensor).cuda(device=self.device)
                curve = curve.type(torch.FloatTensor).cuda(device=self.device)
                coodinates = curve[:,:,:2]
                
                curve = curve.permute(0, 2, 1)
                coodinates = coodinates.permute(0, 2, 1)
                
                # feed into model
                offset = self.model(curve)
                new_coodinates = offset + coodinates
                
                dist1, dist2, _, _ = self.criterion(GT_points, new_coodinates.permute(0,2,1))
                loss = torch.mean(dist1)/self.num_points + torch.mean(dist2)/self.gt_num_points
                
                self.val_loss.add(loss.item())
                self.vis.plot('val_loss', self.val_loss.value()[0], i+epoch*len(self.val_data))
                
            self.model.train()
                
            # vis.plot('val_loss', val_loss_meter.value()[0], epoch)
            # vis.plot('train_loss', train_loss_meter.value()[0], epoch)

                
if __name__=="__main__":
    config = parser()
    trainer = Trainer(config)
    trainer.train()
        
