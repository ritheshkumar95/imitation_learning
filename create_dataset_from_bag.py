import rosbag
import sys
import numpy as np
import os
import h5py
from tqdm import tqdm
from PIL import Image
from scipy.stats import describe
sys.path.append(os.path.join(os.getenv('DUCKIETOWN_ROOT'), 'catkin_ws/src/00-infrastructure/duckietown/include/'))
from duckietown_utils import rgb_from_ros


# bag = rosbag.Bag(sys.argv[1], 'r')
bag = rosbag.Bag('duckduckgo_UdM_2017-12-21-19-45-06.bag', 'r')
image_topic = '/duckduckgo/camera_node/image/compressed'
control_topic = '/duckduckgo/car_cmd_switch_node/cmd'

image_times = []
image_list = []
for msg in bag.read_messages(topics=image_topic):
    image_times.append(msg.timestamp.to_sec())
    image_list.append(msg.message)

control_times = []
control_list = []
for msg in bag.read_messages(topics=control_topic):
    control_times.append(msg.message.header.stamp.to_sec())
    control_list.append([msg.message.v, msg.message.omega])

control_times = np.asarray(control_times)
image_times = np.asarray(image_times)
closest_img_idx = []
delta_t = []
for time in control_times:
    diffs = np.maximum(0, time - image_times)
    idxs = np.nonzero(diffs)[0]
    if idxs.size != 0:
        delta_t.append(diffs[idxs[-1]])
        closest_img_idx.append(idxs[-1])
    else:
        delta_t.append(None)
        closest_img_idx.append(None)

delta_t = np.asarray(delta_t)
print describe(delta_t[np.not_equal(delta_t, None)])


print "Preparing to write to HDF5"
f = h5py.File('data_grayscale.hdf5', 'w')
n_rows = np.not_equal(delta_t, None).sum()
img_data = f.create_dataset(
    "image", (n_rows, ),
    dtype=h5py.special_dtype(vlen=np.dtype('uint8')))
control_data = f.create_dataset(
    "control", (n_rows, ),
    dtype=h5py.special_dtype(vlen=np.dtype('float32')))

count = 0
for i, idx in tqdm(enumerate(closest_img_idx)):
    if idx is None:
        continue

    img = Image.fromarray(rgb_from_ros(image_list[idx]))
    img.thumbnail((80, 60))

    img_data[count] = np.asarray(img.convert('L')).flatten()  # (80, 60, 3)
    control_data[count] = np.asarray(control_list[i]).astype('float32')
    count += 1

f.close()
