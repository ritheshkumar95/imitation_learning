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
<<<<<<< HEAD
    images = tf.random_crop(images, [batch_size, N_IMAGES, 60, 60])
    images = tf.nn.dropout(images, dropout)
    out = tf.nn.leaky_relu(lib.ops.Conv2D(
        'Conv1', images, N_IMAGES, 8, 5, 3
        ), .02)
=======

    images = tf.nn.dropout(images,0.5)#tf.random_normal(shape = tf.shape(images), mean = 0.0, stddev = 0.1, dtype = tf.float32)

    #images = tf.random_crop(images,(batch_size,4,54,72))

    out = tf.nn.leaky_relu(lib.ops.Conv2D(
        'Conv1', images, N_IMAGES*1, 32, 5, 2, batchnorm=True), .02)
>>>>>>> 94027649a771fda14412dfb3f437dfaa542a38bf
    out = tf.nn.leaky_relu(lib.ops.Conv2D(
        'Conv2', out, 8, 8, 5, 2
        ), .02)
    out = tf.nn.leaky_relu(lib.ops.Conv2D(
<<<<<<< HEAD
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
=======
        'Conv3', out, 32, 64, 5, 2, batchnorm=True), .02)
    out = tf.nn.leaky_relu(lib.ops.Conv2D(
        'Conv4', out, 64, 64, 5, 2, batchnorm=True), .02)

    out = tf.nn.leaky_relu(lib.ops.Linear(
        'fc1', tf.reshape(out, (batch_size, -1)), 64 * 5 * 4, 512, batchnorm=False),
        .02
        )

    out = tf.nn.dropout(out,0.5)

    out = lib.ops.Linear('fc2', out, 512, 2)
>>>>>>> 94027649a771fda14412dfb3f437dfaa542a38bf
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
<<<<<<< HEAD
        losses.append([out[0]])
=======

        #out_control = np.random.normal(loc=out[1][:, 0], scale=np.exp(out[1][:, 1]*.5))
        
        out_control = out[1][:, 0]
        # out_control = np.random.normal(loc=out[1][:, 0], scale=1.)
        loss2 = np.mean(np.abs(out_control - batch_control))
        losses.append([out[0], loss2])

>>>>>>> 94027649a771fda14412dfb3f437dfaa542a38bf
        if which_set == 'train' and (i+1) % PRINT_FREQ == 0:
            print "Iter {}: loss - {}".format(i+1, np.asarray(losses).mean(0))

    print "Epoch completed! loss - {}".format(np.asarray(losses).mean(0))
    return np.asarray(losses).mean(0)[0]


dset = Dataset(N_IMAGES-1)
<<<<<<< HEAD

images = tf.placeholder(tf.float32, [None, None, None, None], name="images")
control = tf.placeholder(tf.float32, [None], name="control")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

output = network(images, keep_prob)
loss = tf.reduce_mean(tf.abs(output[:, 0] - control))
=======
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
>>>>>>> 94027649a771fda14412dfb3f437dfaa542a38bf
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
<<<<<<< HEAD
    cost = loop('valid')
    if cost < BEST_COST:
        BEST_COST = cost
        print "Saving Model!"
        saver.save(sess, './best_model.ckpt')
        # saver.export_meta_graph('./best_model.meta')
=======
    loop('valid')


>>>>>>> 94027649a771fda14412dfb3f437dfaa542a38bf
