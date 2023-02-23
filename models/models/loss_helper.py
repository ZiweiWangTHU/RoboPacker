import torch
import torch.nn as nn
from chamfer3D.dist_chamfer_3D import chamfer_3DDist


def flow_reguliarzer(flow):
	'''
	Inputs:
	flow: B M 3
	'''
	return torch.norm(flow, dim=2).mean()


def uncertain_loss(uncertain_logits, CD_distance, threshold=0.1):
	'''
	Inputs:
	uncertain_logits: B 2
	CD_distance: B
	'''
	criterion = nn.CrossEntropyLoss(reduction='none')
	CD_gts = torch.zeros_like(CD_distance) #[16]
	CD_gts[CD_distance < threshold] = 1
	CD_gts = CD_gts.unsqueeze(-1).long().detach()
	return criterion(uncertain_logits, CD_gts).mean()
