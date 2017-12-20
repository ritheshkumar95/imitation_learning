import numpy as np
from data_loader import Dataset
from tqdm import tqdm
import caffe

N_IMAGES = 4

caffe.set_mode_cpu()
dset = Dataset(N_IMAGES - 1)
itr = dset.create_gray_epoch_iterator('valid', 1)
net = caffe.Net('deploy.prototxt', 'deploy.caffemodel', caffe.TEST)

print("Input size: ", net.blobs['data'].data.shape)
print("\nComparisons: ")
losses = []
for i, (batch_img, batch_control) in tqdm(enumerate(itr)):
    net.blobs['data'].data[...] = batch_img
    net.forward()
    output = net.blobs[net.outputs[0]].data.flatten()
    losses.append(abs(output[0] - batch_control[0]))
print("Mean loss on valid set: {}".format(np.mean(losses)))
