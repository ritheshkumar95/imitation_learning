import tflib as lib
import tensorflow as tf
from data_loader import Dataset
import numpy as np
from losses import GaussianNLL
# from losses import StandardGaussianNLL
# from losses import GaussianMixtureNLL


N_IMAGES = 4
BATCH_SIZE = 128
LEARN_RATE = 0.001
PRINT_FREQ = 10
NB_EPOCHS = 50
K = 10


def network(images):
    batch_size = tf.shape(images)[0]

    images = tf.nn.dropout(images,0.5)

    out = tf.nn.leaky_relu(lib.ops.Conv2D(
        'Conv1', images, N_IMAGES, 32, 5, 2, batchnorm=True), .02)
    out = tf.nn.leaky_relu(lib.ops.Conv2D(
        'Conv2', out, 32, 32, 5, 2, batchnorm=True), .02)
    out = tf.nn.leaky_relu(lib.ops.Conv2D(
        'Conv3', out, 32, 64, 5, 2, batchnorm=True), .02)
    out = tf.nn.leaky_relu(lib.ops.Conv2D(
        'Conv4', out, 64, 64, 5, 2, batchnorm=True), .02)
    out = tf.nn.leaky_relu(lib.ops.Linear(
        'fc1', tf.reshape(out, (batch_size, -1)), 64 * 5 * 4, 512, batchnorm=False),
        .02
        )

    out = tf.nn.dropout(out,0.5)

    out = lib.ops.Linear('fc2', out, 512, 2)
    return out


def loop(which_set='train'):
    itr = dset.create_gray_epoch_iterator(which_set, BATCH_SIZE)
    losses = []
    run_vars = [loss, output]
    if which_set == 'train':
        run_vars += [train_op]

    for i, (batch_img, batch_control) in enumerate(itr):
        out = sess.run(run_vars, feed_dict={
            images: batch_img,
            control: batch_control
        })

        #out_control = np.random.normal(loc=out[1][:, 0], scale=np.exp(out[1][:, 1]*.5))
        
        out_control = out[1][:, 0]
        # out_control = np.random.normal(loc=out[1][:, 0], scale=1.)
        loss2 = np.mean(np.abs(out_control - batch_control))
        losses.append([out[0], loss2])

        if which_set == 'train' and (i+1) % PRINT_FREQ == 0:
            print "Iter {}: loss - {}".format(i+1, np.asarray(losses).mean(0))

    print "Epoch completed! loss - {}".format(np.asarray(losses).mean(0))


dset = Dataset(N_IMAGES-1)
images = tf.placeholder(tf.float32, [None, None, None, None])
control = tf.placeholder(tf.float32, [None])

output = network(images)
#loss = tf.reduce_mean(
#    GaussianNLL(control, output[:, 0], output[:, 1])
#)

loss = 1.0 * tf.reduce_mean(tf.abs(output[:,0] - control))

# loss = tf.reduce_mean(
#     StandardGaussianNLL(control, output)
# )
# loss = tf.reduce_mean(
#     GaussianMixtureNLL(control, *tf.split(output, 3, axis=1))
# )

# loss = tf.reduce_mean(tf.abs(output_control - control))
# mse_loss = tf.reduce_mean(tf.square(output_control - control))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARN_RATE)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)

lib.print_params_info()
# raw_input("Waiting..is everything correct?")

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in xrange(NB_EPOCHS):
    print "Epoch %d: " % (i+1)
    print "Training..."
    loop('train')
    print "Validating..."
    loop('valid')


