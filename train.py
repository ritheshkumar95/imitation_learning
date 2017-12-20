import tflib as lib
import tensorflow as tf
from data_loader import Dataset
import numpy as np


N_IMAGES = 4
BATCH_SIZE = 128
LEARN_RATE = 0.001
PRINT_FREQ = 10
NB_EPOCHS = 100


def network(images, dropout):
    batch_size = tf.shape(images)[0]
    images = tf.random_crop(images, [batch_size, N_IMAGES, 60, 60])
    images = tf.nn.dropout(images, dropout)
    out = tf.nn.leaky_relu(lib.ops.Conv2D(
        'Conv1', images, N_IMAGES, 8, 5, 3
        ), .02)
    out = tf.nn.leaky_relu(lib.ops.Conv2D(
        'Conv2', out, 8, 8, 5, 2
        ), .02)
    out = tf.nn.leaky_relu(lib.ops.Conv2D(
        'Conv3', out, 8, 16, 5, 2
        ), .02)
    # out = tf.nn.leaky_relu(lib.ops.Conv2D(
    #     'Conv4', out, 16, 32, 5, 2
    #     ), .02)
    out = tf.nn.leaky_relu(lib.ops.Linear(
            'fc1', tf.reshape(out, (batch_size, -1)), 16 * 5 * 5, 512
        ), .02)
    out = tf.nn.dropout(out, dropout)
    out = lib.ops.Linear('fc2', out, 512, 1)
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
            control: batch_control,
            keep_prob: 0.5 if which_set == 'train' else 1.0
        })
        losses.append([out[0]])
        if which_set == 'train' and (i+1) % PRINT_FREQ == 0:
            print "Iter {}: loss - {}".format(i+1, np.asarray(losses).mean(0))

    print "Epoch completed! loss - {}".format(np.asarray(losses).mean(0))
    return np.asarray(losses).mean(0)[0]


dset = Dataset(N_IMAGES-1)

images = tf.placeholder(tf.float32, [None, None, None, None], name="images")
control = tf.placeholder(tf.float32, [None], name="control")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

output = network(images, keep_prob)
loss = tf.reduce_mean(tf.abs(output[:, 0] - control))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARN_RATE)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)

lib.print_params_info()

tf.add_to_collection("images", images)
tf.add_to_collection("control", control)
tf.add_to_collection("keep_prob", keep_prob)
tf.add_to_collection("output", output)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
sess.run(tf.global_variables_initializer())


BEST_COST = 9999
for i in xrange(NB_EPOCHS):
    print "Epoch %d: " % (i+1)
    print "Training..."
    loop('train')
    print "Validating..."
    cost = loop('valid')
    if cost < BEST_COST:
        BEST_COST = cost
        print "Saving Model!"
        saver.save(sess, './best_model.ckpt')
        # saver.export_meta_graph('./best_model.meta')
