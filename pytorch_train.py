from data_loader import Dataset
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


N_IMAGES = 1
BATCH_SIZE = 128
LEARN_RATE = 0.001
PRINT_FREQ = 10
NB_EPOCHS = 100


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.image_dropout = nn.Dropout(.5)
        self.conv_part = nn.Sequential(
            nn.Conv2d(N_IMAGES, 8, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, 2, 2),
            nn.ReLU(),
            )
        self.fc_part = nn.Sequential(
            nn.Linear(32 * 4 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
            )

    def forward(self, x):
        x = self.image_dropout(x)
        return self.fc_part(self.conv_part(x).view(x.size(0), -1))


def loop(which_set='train'):
    itr = dset.create_gray_epoch_iterator(which_set, BATCH_SIZE)
    losses = []
    for i, (batch_img, batch_control) in enumerate(itr):
        batch_img = Variable(torch.from_numpy(batch_img)).cuda()
        batch_control = Variable(torch.from_numpy(batch_control)).cuda()

        output_control = model(batch_img)
        loss = nn.L1Loss()(output_control.squeeze(), batch_control)
        # loss = nn.CrossEntropyLoss()(
        #     output_control, 
        #     batch_control
        #     )

        if which_set == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()

        losses.append([loss.cpu().data.tolist()[0]])
        if which_set == 'train' and (i + 1) % PRINT_FREQ == 0:
            print "Iter {}: loss - {}".format(
                i + 1, np.asarray(losses).mean(0)
                )

    print "Epoch completed! loss - {}".format(np.asarray(losses).mean(0))
    return np.asarray(losses).mean(0)[0]


dset = Dataset(N_IMAGES - 1)
model = Network().cuda()
opt = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

BEST_COST = 9999
for i in xrange(NB_EPOCHS):
    print "Epoch %d: " % (i + 1)
    print "Training..."
    model.train()
    loop('train')
    print "Validating..."
    # model.eval()
    cost = loop('valid')
    if cost < BEST_COST:
        BEST_COST = cost
        print "Saving Model!"
        torch.save(model.state_dict(), 'best_model.torch')
    else:
        print "Not saving! Best model loss - {}".format(BEST_COST)
