import os
import numpy as np
import cv2
import torch
from models2 import reinforcement_net
from scipy import ndimage
from tester import Tester
import time
import datetime


class Trainer(object):
    def __init__(self, force_cpu):

        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

        # Fully convolutional Q network for deep reinforcement learning
        self.model = reinforcement_net(self.use_cuda)

        # Initialize Huber loss
        # self.criterion = torch.nn.SmoothL1Loss(reduction='none')  # Huber loss
        self.criterion = torch.nn.L1Loss()

        if self.use_cuda:
            self.criterion = self.criterion.cuda()

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()

        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=2e-4)

        self.iteration = 0


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


    def backprop(self, output, labels):

        # Compute loss and backward pass
        self.optimizer.zero_grad()
        if self.use_cuda:
            loss = self.criterion(output, torch.Tensor(labels).float().squeeze().cuda())
        else:
            loss = self.criterion(output, torch.Tensor(labels).float().squeeze())

        loss.backward()
        loss_value = loss.cpu().detach().numpy()
        self.optimizer.step()

        return loss_value


import tqdm
from sklearn.model_selection import train_test_split
import torch.utils.data as Data

BATCH_SIZE = 8
trainer = Trainer(force_cpu=False)

dataset = 'median'
if dataset == 'large':
    data_path = 'training_data/grasp_train_data/large'
    num_samples = 10000
elif dataset == 'median':
    data_path = 'training_data/grasp_train_data/median'
    num_samples = 4400
elif dataset == 'mini':
    data_path = 'training_data/grasp_train_data/mini'
    num_samples = 1000
elif dataset == '0516':
    data_path = 'training_data/grasp_train_data/0516'
    num_samples = 1010

depth_dataset = []
target_dataset = []
grasp_dataset = []
for filename in tqdm.tqdm(sorted(os.listdir(os.path.join(data_path, 'depth')))):
    depth_data = cv2.imread(os.path.join(os.path.join(data_path, 'depth'), filename), cv2.IMREAD_GRAYSCALE)
    depth_dataset.append(depth_data)
for filename in tqdm.tqdm(sorted(os.listdir(os.path.join(data_path, 'target')))):
    target_data = cv2.imread(os.path.join(os.path.join(data_path, 'target'), filename), cv2.IMREAD_GRAYSCALE)
    target_dataset.append(target_data)
for filename in tqdm.tqdm(sorted(os.listdir(os.path.join(data_path, 'grasp')))):
    grasp_data = cv2.imread(os.path.join(os.path.join(data_path, 'grasp'), filename), cv2.IMREAD_GRAYSCALE)
    grasp_dataset.append(grasp_data)

depth_dataset = np.asarray(depth_dataset, dtype=np.float32)
target_dataset = np.asarray(target_dataset, dtype=np.float32)
grasp_dataset = np.asarray(grasp_dataset, dtype=np.float32)
dataset = np.stack((depth_dataset, target_dataset, grasp_dataset), axis=1)

labels = []
with open(os.path.join(data_path, 'labels.txt'), 'r') as f:
    for _ in range(num_samples):
        value = f.readline().strip('\n')
        labels.append(float(value))
labels = np.asarray(labels)

train_data, test_data, train_labels, test_labels = train_test_split(dataset, labels, test_size=0.20, random_state=4)
train_data, test_data, train_labels, test_labels = torch.FloatTensor(train_data), torch.FloatTensor(test_data), torch.FloatTensor(train_labels), torch.FloatTensor(test_labels)
train_dataset = Data.TensorDataset(train_data, train_labels)
test_dataset = Data.TensorDataset(test_data, test_labels)
num_train = len(train_labels)
num_test = len(test_labels)

train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True)

timestamp = time.time()
timestamp_value = datetime.datetime.fromtimestamp(timestamp)
logging_directory = os.path.join(os.path.abspath('record'), timestamp_value.strftime('%Y-%m-%d.%H:%M:%S'))
if not os.path.exists(logging_directory):
    os.mkdir(logging_directory)
for epoch in range(50):
    loss_sum = 0.0
    trainer.model.train()
    if epoch >= 2:
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-4)

    for i, (batch_data, batch_label) in enumerate(train_loader):
        batch_output = torch.Tensor([]).cuda()
        for sample in batch_data:
            depth_heightmap, target_mask, grasp_mask = sample[0], sample[1], sample[2]
            conf, _ = trainer.forward(depth_heightmap, target_mask, grasp_mask)
            batch_output = torch.cat((batch_output, conf.unsqueeze(0)))

        loss_value = trainer.backprop(batch_output, batch_label)
        loss_sum += loss_value
        # print('| Epoch: ', epoch, '| Batch: ', i, '| loss: %.04f'%(loss_value / BATCH_SIZE))
        print('| Epoch: ', epoch, '| Batch: ', i, '| loss: %.04f'%(loss_value))
    loss_avg_epoch = loss_sum / num_train * BATCH_SIZE
    print('Epoch average loss: ', loss_avg_epoch)

    with open('%s/loss_rec.txt'%logging_directory, mode='a') as f:
        f.write(str(loss_avg_epoch)+'\n')
        f.close()

    # testing
    with torch.no_grad():
        if (epoch+1) % 1 == 0:
            if epoch >= 1:
                torch.save(trainer.model.state_dict(), 'saved_models2/grasp/%03d.pkl'%(epoch+1))

            trainer.model.eval()
            tester = Tester(model=trainer.model)
            print('testing...')
            correct, total = 0, 0
            loss_sum = 0.0
            for batch_data, batch_label in test_loader:
                batch_output = torch.Tensor([]).cuda()
                for sample in batch_data:
                    depth_heightmap, target_mask, grasp_mask = sample[0], sample[1], sample[2]
                    conf, _ = tester.forward(depth_heightmap, target_mask, grasp_mask) #
                    batch_output = torch.cat((batch_output, conf.unsqueeze(0)))
                print('*************************************')
                print('batch_output: ', batch_output.detach().cpu().numpy())
                print('batch_labels: ', batch_label)
                loss = tester.criterion(batch_output, torch.Tensor(batch_label).float().squeeze().cuda())
                loss = loss.sum()
                loss_value = loss.cpu().detach().numpy()
                loss_sum += loss_value
                # Compute prediction accuracy
                predicted = batch_output
                predicted[batch_output < 0.5] = 0.0
                predicted[(batch_output > 0.5) & (batch_output < 1.5)] = 1.0
                predicted[batch_output > 1.5] = 2.0
                # binary classification
                # predicted[batch_output < 1.0] = 0.0
                # predicted[batch_output > 1.0] = 2.0

                print('predic_label: : ', predicted)
                total += batch_label.size(0)
                correct += (predicted == batch_label.cuda()).sum().item()
            accuracy = correct / total
            print('Accuracy of the network on the test images: %d %%' % (100 * accuracy))
            loss_avg_test = loss_sum / num_test * BATCH_SIZE
            print('TEST average loss: ', loss_avg_test)

            f = open('%s/loss_rec_test.txt'%logging_directory, mode='a')
            f.write(str(loss_avg_test) + ' ' + str(accuracy) + '\n')
            f.close()





