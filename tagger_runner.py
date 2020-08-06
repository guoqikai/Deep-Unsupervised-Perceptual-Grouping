import tensorflow as tf
import numpy as np
from tagger import Tagger
import data

num_examples = 60000
num_epochs = 20
num_labeled = 100

starter_learning_rate = 0.02

decay_after = 15

batch_size = 2000

num_iter = int((num_examples/batch_size) * num_epochs)

attr = {}
attr["iterations"] = 3
attr["num_groups"] = 3
attr["input_noise"] = 0.2
attr["ladder_layer_sizes"] = [3000, 2000, 1000]
attr["input_size"] = 784
attr["init_z_val"] = 0.2

if __name__ == "__main__":
    t = Tagger(**attr)
    learning_rate = tf.Variable(starter_learning_rate, trainable=False)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(t.cost)

    mnist = data.read_data_sets("MNIST_data", n_labeled=num_labeled, one_hot=True)
    saver = tf.train.Saver()

    print("===  Starting Session ===")
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    print("=== Training ===")

    for i in range(num_iter):
        print("batch:", i % (num_examples // batch_size), " epoch:", i // (num_examples // batch_size))
        images, labels = mnist.train.next_batch (batch_size)
        sess.run(train_step, feed_dict={t.inputs_unlabeled: images, t.inputs_labeled: np.zeros(shape=(0, 784)), t.targets_labeled: [[0]]})
        if (i > 1) and ((i + 1) % (num_iter / num_epochs) == 0):
            epoch_n = i // (num_examples // batch_size)
            if (epoch_n + 1) >= decay_after:
                # decay learning rate
                # learning_rate = starter_learning_rate * ((num_epochs - epoch_n) / (num_epochs - decay_after))
                ratio = 1.0 * (num_epochs - (epoch_n + 1))  # epoch_n + 1 because learning rate is set for next epoch
                ratio = max (0, ratio / (num_epochs - decay_after))
                sess.run(learning_rate.assign(starter_learning_rate * ratio))
