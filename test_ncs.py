import numpy as np
from data_loader import Dataset
from mvnc import mvncapi as mvnc
import time
from tqdm import tqdm

N_IMAGES = 4

dset = Dataset(N_IMAGES - 1)
itr = dset.create_gray_epoch_iterator('valid', 1)

mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print("No devices found!")
    quit()
device = mvnc.Device(devices[0])
device.OpenDevice()
with open('graph', mode='rb') as f:
    blob = f.read()
graph = device.AllocateGraph(blob)

print("\nComparisons: ")
# for i in range(30):
#     batch_img, batch_control = itr.__next__()
#     start = time.time()
#     graph.LoadTensor(batch_img.astype(np.float16), 'user object')
#     output, userobj = graph.GetResult()
#     end = time.time()
#     print("{}.) pred: {} actual: {} time: {}".format(
#         i+1, output, batch_control, end-start)
#         )
losses = []
for i, (batch_img, batch_control) in tqdm(enumerate(itr)):
    batch_img = batch_img[0].transpose(1, 2, 0)
    print(batch_img.shape)
    out = graph.LoadTensor(batch_img.astype(np.float16), 'user object')
    output, userobj = graph.GetResult()
    losses.append(abs(output[0] - batch_control.astype(np.float16)[0]))

print("Mean loss on valid set: {}".format(np.mean(losses)))

graph.DeallocateGraph()
device.CloseDevice()
