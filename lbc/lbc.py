import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from utils import _numpy
from .models import PointModel, RGBPointModel, Converter

class LBC:
    def __init__(self, args):
        
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        for key, value in config.items():
            setattr(self, key, value)
        
        self.crop_size = 64
        self.num_cmds = 6

        # Save configs
        self.device = torch.device(args.device)
        self.T = self.num_plan # T in LBC
        
        # Create models
        self.bev_model = PointModel(
            'resnet18',
            height=64, width=64,
            input_channel=12,
            output_channel=self.T*self.num_cmds
        ).to(self.device)
        
        self.rgb_model = RGBPointModel(
            'resnet34',
            pretrained=True,
            height=240-self.crop_top-self.crop_bottom, width=480,
            output_channel=self.T*self.num_cmds
        ).to(self.device)
        
        self.bev_optim = optim.Adam(self.bev_model.parameters(), lr=args.lr)
        self.rgb_optim = optim.Adam(self.rgb_model.parameters(), lr=args.lr)
        
        self.converter = Converter(offset=6.0, scale=[1.5,1.5]).to(self.device)
        
    def train(self, rgbs, lbls, sems, locs, spds, cmds, train='image'):
        
        rgbs = rgbs.permute(0,3,1,2).float().to(self.device)
        lbls = lbls.permute(0,3,1,2).float().to(self.device)
        sems = sems.long().to(self.device)
        locs = locs.float().to(self.device)
        spds = spds.float().to(self.device)
        cmds = cmds.long().to(self.device)
        
        if train == 'bev':
            return self.train_bev(lbls, spds, locs, cmds)
            
        elif train == 'rgb':
            return self.train_rgb(rgbs, lbls, sems, spds, cmds)
        
        else:
            raise NotImplementedError
            
    def train_bev(self, lbls, spds, locs, cmds):
        
        pred_locs = self.bev_model(lbls, spds).view(-1,self.num_cmds,self.T,2)

        # Scale pred locs
        pred_locs = (pred_locs+1) * self.crop_size/2

        loss = F.mse_loss(pred_locs.gather(1, cmds[:,None,None,None].repeat(1,1,self.T,2)).squeeze(1), locs)

        self.bev_optim.zero_grad()
        loss.backward()
        self.bev_optim.step()
        
        return dict(
            loss=float(loss),
            cmds=_numpy(cmds),
            locs=_numpy(locs),
            pred_locs=_numpy(pred_locs),
        )
    
    def train_rgb(self, rgbs, lbls, sems, spds, cmds):
        
        with torch.no_grad():
            tgt_bev_locs = (self.bev_model(lbls, spds).view(-1,self.num_cmds,self.T,2)+1) * self.crop_size/2
        
        pred_rgb_locs, pred_sems = self.rgb_model(rgbs, spds)
        pred_rgb_locs = (pred_rgb_locs.view(-1,self.num_cmds,self.T,2)+1) * self.rgb_model.img_size/2
    
        tgt_rgb_locs = self.bev_to_cam(tgt_bev_locs)
        pred_bev_locs = self.cam_to_bev(pred_rgb_locs)
        
        act_loss = F.l1_loss(pred_bev_locs, tgt_bev_locs, reduction='none').mean(dim=[2,3])
        
        turn_loss = (act_loss[:,0]+act_loss[:,1]+act_loss[:,2]+act_loss[:,3])/4
        lane_loss = (act_loss[:,4]+act_loss[:,5]+act_loss[:,3])/3
        foll_loss = act_loss[:,3]
        
        is_turn = (cmds==0)|(cmds==1)|(cmds==2)
        is_lane = (cmds==4)|(cmds==5)

        loc_loss = torch.mean(torch.where(is_turn, turn_loss, foll_loss) + torch.where(is_lane, lane_loss, foll_loss))
        
        # multip_branch_losses = losses.mean(dim=[1,2,3])
        # single_branch_losses = losses.mean(dim=[2,3]).gather(1, cmds[:,None]).mean(dim=1)
        
        # loc_loss = torch.where(cmds==3, single_branch_losses, multip_branch_losses).mean()
        seg_loss = F.cross_entropy(F.interpolate(pred_sems,scale_factor=4), sems)
        
        loss = loc_loss + self.seg_weight * seg_loss
        
        self.rgb_optim.zero_grad()
        loss.backward()
        self.rgb_optim.step()
        
        return dict(
            loc_loss=float(loc_loss),
            seg_loss=float(seg_loss),
            cmds=_numpy(cmds),
            tgt_rgb_locs=_numpy(tgt_rgb_locs),
            tgt_bev_locs=_numpy(tgt_bev_locs),
            pred_rgb_locs=_numpy(pred_rgb_locs),
            pred_bev_locs=_numpy(pred_bev_locs),
            tgt_sems=_numpy(sems),
            pred_sems=_numpy(pred_sems.argmax(1)),
        )


    def bev_to_cam(self, bev_coords):
        
        bev_coords = bev_coords.clone()
        bev_coords[...,1] =  self.crop_size/2 - bev_coords[...,1]
        bev_coords[...,0] = -self.crop_size/2 + bev_coords[...,0]
        world_coords = torch.flip(bev_coords, [-1])
        
        cam_coords = self.converter.world_to_cam(world_coords)
        
        return cam_coords
        
    def cam_to_bev(self, cam_coords):
        world_coords = self.converter.cam_to_world(cam_coords)
        bev_coords = torch.flip(world_coords, [-1])
        bev_coords[...,1] *= -1
        
        return bev_coords + self.crop_size/2
