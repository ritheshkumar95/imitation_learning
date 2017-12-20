import h5py
import numpy as np
#from torchvision.utils import save_image
#import torch


class Dataset(object):
    def __init__(self, n_back_images=3):
        self.hdf5 = h5py.File('./data/data_grayscale.hdf5', 'r')
        self.n_back_images = n_back_images
        self.len_data = len(self.hdf5['image'])
        self.idxs = {}
        idxs = np.arange(n_back_images, self.len_data)
        np.random.seed(111)
        np.random.shuffle(idxs)
        self.idxs['train'] = idxs[:int(0.9 * self.len_data)]
        self.idxs['valid'] = idxs[int(0.9 * self.len_data):]

    def create_rgb_epoch_iterator(self, which_set='train', batch_size=32):
        idxs = self.idxs[which_set]
        for i in range(0, len(idxs), batch_size):
            batch_idxs = idxs[i: i+batch_size]
            batch_images = []
            batch_control = []
            for idx in batch_idxs:
                batch_images.append(
                    np.stack(
                        self.hdf5['image'][idx-self.n_back_images:idx+1], 0
                        ).reshape(self.n_back_images+1, 60, 80, 3)
                    )
                batch_control.append(self.hdf5['control'][idx][1])

            batch_images = np.stack(
                batch_images, 0
                ).transpose(0, 1, 4, 2, 3).reshape(-1, 3*(self.n_back_images+1), 60, 80)
            batch_control = np.asarray(batch_control)

            yield (2*(batch_images[:, :, :, 10:-10]/255.)-1.).astype('float32'), batch_control.astype('float32')

    def create_gray_epoch_iterator(self, which_set='train', batch_size=32):
        idxs = self.idxs[which_set]
        for i in range(0, len(idxs), batch_size):
            batch_idxs = idxs[i: i+batch_size]
            batch_images = []
            batch_control = []
            for idx in batch_idxs:
                batch_images.append(
                    np.stack(
                        self.hdf5['image'][idx-self.n_back_images:idx+1], 0
                        ).reshape(self.n_back_images+1, 60, 80)
                    )
                batch_control.append(self.hdf5['control'][idx][1])

            batch_images = np.stack(batch_images, 0)
            batch_control = np.asarray(batch_control)
            yield (2*(batch_images/255.)-1.).astype('float32'), batch_control.astype('float32')


if __name__ == '__main__':
    dset = Dataset()
    itr = dset.create_gray_epoch_iterator(batch_size=16)
    images, control = itr.next()
    # images = (images.reshape(-1, 3, 60, 80)+1)/2.
    images = (images.reshape(-1, 1, 60, 80)+1)/2.
    save_image(torch.from_numpy(images), 'test.png', nrow=8)
