import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation, PointNetSetAbstractionMsg


class DeformFlowNet(nn.Module):
    def __init__(self, additional_channel=0, Is_MSG=False):
        super(DeformFlowNet, self).__init__()

        self.partial_encoder = PointNetEncoder(global_feat=True, spatial_transform=True, feature_transform=False, channel=3+additional_channel)
        if Is_MSG:
            self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
            self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
            self.sa3 = PointNetSetAbstraction(32, radius=1.6, nsample=128, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=False)
            self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
            self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
            self.fp1 = PointNetFeaturePropagation(in_channel=150 + 1008, mlp=[128, 128])
        else:
            self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6+additional_channel, mlp=[64, 64, 128], group_all=False)
            self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
            self.sa3 = PointNetSetAbstraction(npoint=32, radius=0.8, nsample=128, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=False)
            self.fp3 = PointNetFeaturePropagation(in_channel=2307, mlp=[256, 256])
            self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
            #self.fp1 = PointNetFeaturePropagation(in_channel=128+16+6+additional_channel, mlp=[128, 128, 128])
            self.fp1 = PointNetFeaturePropagation(in_channel=128 + 3, mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 3, 1)
        
        self.conv_uncertain = nn.Conv1d(2048, 128, 1)
        self.bn_uncertain = nn.BatchNorm1d(128)
        self.drop_uncertain = nn.Dropout(0.5)
        self.conv_uncertain2 = nn.Conv1d(128, 2, 1)

    def forward(self, xyz_template, xyz_partial):
        '''
        Inputs:
        xyz_template: B 3+additional_channel M (M>512)
        xyz_partial: B 3+additional_channel N

        Outputs:
        xyz_deform_template: B 3 M
        flow: B 3 M
        '''
        # Set Abstraction layers
        B, C, M = xyz_template.shape
        l0_points = xyz_template
        l0_xyz = xyz_template[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        partial_feats = self.partial_encoder(xyz_partial)[0]
        # Feature Propagation layers
        partial_feats_rep = partial_feats.unsqueeze(-1).repeat(1, 1, l2_xyz.shape[2])  # B 1024 --> B 1024 M
        l2_points = self.fp3(l2_xyz, l3_xyz, torch.cat([partial_feats_rep,l2_xyz,l2_points],1), l3_points)
        #l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # partial_feats, _, _ = self.partial_encoder(xyz_partial).unsqueeze(-1).repeat(1, 1, M) # B 1024 --> B 1024 M
        # partial_feats_rep = partial_feats.unsqueeze(-1).repeat(1, 1, M)  # B 1024 --> B 1024 M
        # l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([partial_feats_rep,l0_xyz,l0_points],1), l1_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        flow = self.conv2(x)
        uncertain_logits = F.relu(self.bn_uncertain(self.conv_uncertain(torch.cat([partial_feats.unsqueeze(-1).detach(), l3_points.max(-1,keepdim=True)[0].detach()], dim=1))))
        uncertain_logits = self.drop_uncertain(uncertain_logits)
        uncertain_logits = self.conv_uncertain2(uncertain_logits)
        return flow, uncertain_logits
