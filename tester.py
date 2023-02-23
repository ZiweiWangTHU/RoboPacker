import numpy as np
import torch
from scipy import ndimage


class Tester(object):
    def __init__(self, model):

        # Check if CUDA can be used
        if torch.cuda.is_available():
            self.use_cuda = True

        # Fully convolutional Q network for deep reinforcement learning
        self.model = model

        # Initialize Huber loss
        # self.criterion = torch.nn.SmoothL1Loss(reduction='none')  # Huber loss
        self.criterion = torch.nn.L1Loss()
        # self.criterion = torch.nn.CrossEntropyLoss()


        if self.use_cuda:
            self.criterion = self.criterion.cuda()

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()

        # # Set model to evaluating mode
        # self.model.eval()

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-5, momentum=0.9, weight_decay=2e-5)

    # Compute forward pass through model to compute affordances/Q
    def forward(self, depth_heightmap, target_mask_heightmap, grasp_mask_heightmap):

        # Apply 2x scale to input heightmaps
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2, 2], order=0)
        target_mask_heightmap_2x = ndimage.zoom(target_mask_heightmap, zoom=[2, 2], order=0)
        grasp_mask_heightmap_2x = ndimage.zoom(grasp_mask_heightmap, zoom=[2, 2], order=0)
        assert (depth_heightmap_2x.shape[0:2] == target_mask_heightmap_2x.shape[0:2])

        # Add extra padding (to handle rotations inside network)
        diag_length = float(depth_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - depth_heightmap_2x.shape[0]) / 2)
        depth_heightmap_2x = np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)
        target_mask_heightmap_2x = np.pad(target_mask_heightmap_2x, padding_width, 'constant', constant_values=0)
        grasp_mask_heightmap_2x = np.pad(grasp_mask_heightmap_2x, padding_width, 'constant', constant_values=0)

        # Pre-process input images
        image_mean = 0.01
        image_std = 0.03
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = (depth_heightmap_2x - image_mean) / image_std

        target_mask_heightmap_2x.shape = (target_mask_heightmap_2x.shape[0], target_mask_heightmap_2x.shape[1], 1)
        input_target_mask_image = target_mask_heightmap_2x

        grasp_mask_heightmap_2x.shape = (grasp_mask_heightmap_2x.shape[0], grasp_mask_heightmap_2x.shape[1], 1)
        input_grasp_mask_image = grasp_mask_heightmap_2x

        # Construct minibatch of size 1 (b,c,h,w)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_target_mask_image.shape = (input_target_mask_image.shape[0], input_target_mask_image.shape[1], input_target_mask_image.shape[2], 1)
        input_grasp_mask_image.shape = (input_grasp_mask_image.shape[0], input_grasp_mask_image.shape[1], input_grasp_mask_image.shape[2], 1)

        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3, 2, 0, 1)
        input_target_mask_data = torch.from_numpy(input_target_mask_image.astype(np.float32)).permute(3, 2, 0, 1)
        input_grasp_mask_data = torch.from_numpy(input_grasp_mask_image.astype(np.float32)).permute(3, 2, 0, 1)

        # Pass input data through model
        confidence, state_feat = self.model.forward(input_depth_data, input_target_mask_data, input_grasp_mask_data)

        return confidence, state_feat





