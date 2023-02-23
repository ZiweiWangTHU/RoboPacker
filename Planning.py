
import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import Sequence_net, Placement_net
from scipy import ndimage
import matplotlib.pyplot as plt

class PlanModule(object):
    def __init__(self, resolution, TopHeight, use_cuda = False):

        self.resolution = resolution
        self.TopHeight = TopHeight
        self.use_cuda = use_cuda
        pitch_rolls = np.array([[0, 0],[0, np.pi/2],[0, np.pi],[0, 3*np.pi/2],[np.pi/2, 0],[3*np.pi/2, 0]])
        self.transforms = []
        for pitch_roll in pitch_rolls:
            self.transforms.append(np.concatenate((np.repeat([pitch_roll], 4, axis=0).T, [np.arange(0, 2*np.pi, np.pi/2)]),axis=0).T)
        self.transforms = np.array(self.transforms)
        self.transforms.shape = (24,3)
        # Fully convolutional Q network for deep reinforcement learning
        self.SequenceModel = Sequence_net(self.use_cuda)
        self.PlacementModel = Placement_net(self.use_cuda, self.transforms)
        self.SequenceModel.load_state_dict(torch.load('./models/suquence_model.pth'))
        self.PlacementModel.load_state_dict(torch.load('./models/placement_model.pth'))
        if use_cuda:
            self.SequenceModel.cuda()
            self.PlacementModel.cuda()
        
        # Initialize optimizer
        self.Sequence_optimizer = torch.optim.Adam(self.PlacementModel.parameters(), lr=1e-3, weight_decay=2e-5)
        self.Placement_optimizer = torch.optim.Adam(self.PlacementModel.parameters(), lr=1e-3, weight_decay=2e-5)
        self.iteration = 0

        # Initialize lists to save execution info and RL variables
        self.executed_action_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.predicted_value_log = []
        self.use_heuristic_log = []
        self.is_exploit_log = []
        self.clearance_log = []


    # Pre-load execution info and RL variables
    def preload(self, transitions_directory):
        self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
        self.iteration = self.executed_action_log.shape[0] - 2
        self.executed_action_log = self.executed_action_log[0:self.iteration,:]
        self.executed_action_log = self.executed_action_log.tolist()
        self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
        self.label_value_log = self.label_value_log[0:self.iteration]
        self.label_value_log.shape = (self.iteration,1)
        self.label_value_log = self.label_value_log.tolist()
        self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'), delimiter=' ')
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        self.predicted_value_log.shape = (self.iteration,1)
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
        self.reward_value_log = self.reward_value_log[0:self.iteration]
        self.reward_value_log.shape = (self.iteration,1)
        self.reward_value_log = self.reward_value_log.tolist()
        self.use_heuristic_log = np.loadtxt(os.path.join(transitions_directory, 'use-heuristic.log.txt'), delimiter=' ')
        self.use_heuristic_log = self.use_heuristic_log[0:self.iteration]
        self.use_heuristic_log.shape = (self.iteration,1)
        self.use_heuristic_log = self.use_heuristic_log.tolist()
        self.is_exploit_log = np.loadtxt(os.path.join(transitions_directory, 'is-exploit.log.txt'), delimiter=' ')
        self.is_exploit_log = self.is_exploit_log[0:self.iteration]
        self.is_exploit_log.shape = (self.iteration,1)
        self.is_exploit_log = self.is_exploit_log.tolist()
        self.clearance_log = np.loadtxt(os.path.join(transitions_directory, 'clearance.log.txt'), delimiter=' ')
        self.clearance_log.shape = (self.clearance_log.shape[0],1)
        self.clearance_log = self.clearance_log.tolist()

    def sequence(self, items, templates, uncertainty, items_Hts,  Hbox):
        items_Hts_padding = []
        uncertain = []
        items = np.array(items)
        order = self.pre_order(items_Hts, templates)
        ht_requires = [0,8,4,12,16,20]
        for i in range(min(len(order),20)):
            Ht_padding = []
            for require in ht_requires:
                item_Ht = items_Hts[order[i]][require]
                w, h = item_Ht.shape
                pad1, pad2 = int(np.floor((self.resolution-w)/2)), int(np.ceil((self.resolution-w)/2))
                pad3, pad4 = int(np.floor((self.resolution-h)/2)), int(np.ceil((self.resolution-h)/2))
                Ht_padding.append(np.pad(item_Ht, ((pad1,pad2),(pad3,pad4)), 'constant', constant_values=0))
                if len(Ht_padding) == 2:
                    Ht_padding.append(Hbox)
                    items_Hts_padding.append(np.array(Ht_padding))
                    Ht_padding = []
                    uncertain.append(uncertainty[i])
        items_Hts_padding = np.array(items_Hts_padding)
        input_Ht = torch.from_numpy(items_Hts_padding.astype(np.float32))
        output_prob = self.SequenceModel.forward(input_Ht, items_Hts, templates, uncertain)
        sub_order = np.argsort(output_prob)[::-1]
        order = order[sub_order]
        return order
    
    def placement(self, template, item_Hts, item_Hbs, Hbox):
        inputs_padding = []
        for i in range(len(item_Hts)):
            item_Ht, item_Hb = item_Hts[i], item_Hbs[i]
            if item_Ht.shape!=item_Hb.shape:
                print("Ht and Hb shape error!",item_Ht.shape, item_Hb.shape)
            input_padding = []
            w, h = item_Ht.shape
            #print(w,h)
            pad1, pad2 = int(np.floor((self.resolution-w)/2)), int(np.ceil((self.resolution-w)/2))
            pad3, pad4 = int(np.floor((self.resolution-h)/2)), int(np.ceil((self.resolution-h)/2))
            input_padding.append(np.pad(item_Ht, ((pad1,pad2),(pad3,pad4)), 'constant', constant_values=0))
            input_padding.append(np.pad(item_Hb, ((pad1,pad2),(pad3,pad4)), 'constant', constant_values=self.TopHeight))
            input_padding.append(Hbox)
            inputs_padding.append(np.array(input_padding))
        inputs_padding = np.array(inputs_padding)
        inputs_padding = torch.from_numpy(inputs_padding.astype(np.float32))
        output_mat = self.PlacementModel.forward(inputs_padding, template, item_Hts, item_Hbs, Hbox)
        trans,_,x,y = np.unravel_index(np.argmax(output_mat), output_mat.shape)
        w, h = item_Hts[trans].shape
        x_center = np.round(x + w/2)
        y_center = np.round(y + h/2)
        Z = np.max(Hbox[x:x+w, y:y+h]-item_Hbs[trans])
        euler = self.transforms[trans]
        return x,y,x_center, y_center, Z, euler
    
    def pre_order(self, items_Hts, template):
        volume = []
        for i in range(len(template)):
            volume.append(items_Hts[i][0].size * items_Hts[i][5].shape[1])
            if template[i] == 'banana':
                volume[-1] = volume[-1]*0.4
        pre_order = np.argsort(volume)[::-1]
        return pre_order