from data_loader import Dataset
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from pytorch2caffe import pytorch2caffe
from visualize import make_dot


N_IMAGES = 4
BATCH_SIZE = 128
LEARN_RATE = 0.001
PRINT_FREQ = 10
NB_EPOCHS = 100


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv_part = nn.Sequential(
            nn.Conv2d(N_IMAGES, 8, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, 2, 2),
            nn.ReLU()
        )
        self.fc_part = nn.Sequential(
            nn.Linear(32 * 4 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.fc_part(self.conv_part(x).view(x.size(0), -1))


dset = Dataset(N_IMAGES - 1)
model = Network().cuda()
model.eval()
model.load_state_dict(torch.load('best_model.torch'))
model = model.cpu()

itr = dset.create_gray_epoch_iterator('train', BATCH_SIZE)
losses = []
batch_img, batch_control = itr.next()
batch_img = Variable(torch.from_numpy(batch_img))
output_var = model(batch_img)

fp = open("out.dot", "w")
dot = make_dot(output_var)
print >> fp, dot
fp.close()

pytorch2caffe(batch_img, output_var, 'deploy.prototxt', 'deploy.caffemodel')
