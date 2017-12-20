import tensorflow as tf
from data_loader import Dataset


N_IMAGES = 4

dset = Dataset(N_IMAGES - 1)
itr = dset.create_gray_epoch_iterator('valid', 8)

sess = tf.Session()
saver = tf.train.import_meta_graph('./best_model.ckpt.meta')
saver.restore(sess, './best_model.ckpt')

images = tf.get_collection("images")[0]
control = tf.get_collection("control")[0]
keep_prob = tf.get_collection("keep_prob")[0]
bn_mode = tf.get_collection("batchnorm_mode")[0]
output = tf.get_collection("output")

batch_images, batch_control = itr.next()
output_control = sess.run(output, feed_dict={
    images: batch_images,
    control: batch_control,
    keep_prob: 1.0,
    bn_mode: False
})[0].flatten().tolist()

print "\nComparisons: "
for i, (x, y) in enumerate(zip(output_control, batch_control)):
    print "{}.) pred: {} actual: {}".format(i, x, y)
