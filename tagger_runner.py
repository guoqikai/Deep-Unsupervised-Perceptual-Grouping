import tensorflow as tf
import numpy as np
import os
from PIL import Image

from tagger import Tagger
import data_overlapped

num_examples = 30000
num_epochs = 300
num_labeled = 0
starter_learning_rate = 0.02
batch_size = 300
test_batch_size = 100

num_iter = int((num_examples/batch_size) * num_epochs)

attr = {}
attr["iterations"] = 4
attr["num_groups"] = 5
attr["noise_std"] = 0.2
attr["ladder_layer_sizes"] = [1500, 1000, 500]
attr["input_size"] = 784
attr["init_z_val"] = 0.5

# Set it to > 0 will enable semi supervised training, don't forget to also set num_labeled.
attr["class_cost"] = 0.
attr["batch_size"] = batch_size


def to_img(m, img_shape):
    mi = np.min(m)
    ma = np.max(m)
    return np.reshape(np.uint8((m - mi) * (1 / (ma-mi)) * 255), img_shape)


def save_test(z, m, test, dir_):
    z = np.array(z)
    m = np.array(m)
    for i in range(z.shape[0]):
        test_path = dir_ + "/" + str(i) + "th_iter"
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        for j in range(z.shape[2]):
            sample_path = test_path + "/sample" + str(j)
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            n_test = Image.fromarray(to_img(test[j], (28, 28)), "L")
            n_test.save(sample_path + "/input.jpg")
            for n in range(z.shape[1]):
                n_m = Image.fromarray(to_img(m[i][n][j], (28, 28)), "L")
                n_z = Image.fromarray(to_img(z[i][n][j], (28, 28)), "L")
                n_m.save(sample_path + "/m_group" + str(n) + ".jpg")
                n_z.save(sample_path + "/z_group" + str(n) + ".jpg")


def test():
    test, test_labels = mnist.test.next_batch (test_batch_size)
    z, m, c = sess.run([tagger.z, tagger.m, tagger.cost], feed_dict={
        tagger.inputs_unlabeled: test[:-1],
        tagger.inputs_labeled: np.zeros((0, 784)),
        tagger.targets_labeled: np.zeros((0, 10))
    })

    # for semi-supervised training use the following code
    # z, m, c = sess.run ([tagger.z, tagger.m, tagger.cost], feed_dict={
    #     tagger.inputs_unlabeled: test[:-1],
    #     tagger.inputs_labeled: test[-1:],
    #     tagger.targets_labeled: np.zeros((1, 10))
    # })
    print("cost:", c, "|| exporting result for " + str (test_batch_size) + " test samples...")
    save_test(z, m, test, "test")


def train(i, b_size, tb_size):
    try:
        print("=== Training ===")
        while i < num_iter:
            print("batch:", i % (num_examples // batch_size), " epoch:", i // (num_examples // batch_size))
            img, train_labels = mnist.train.next_batch(batch_size)

            sess.run(train_step, feed_dict={
                tagger.inputs_unlabeled: img,
                tagger.inputs_labeled: img[:num_labeled],
                tagger.targets_labeled: train_labels
            })
            if (i > 1) and ((i + 1) % (num_iter / num_epochs) == 0):
                saver.save(sess, 'checkpoints/model.ckpt', global_step=i // (num_examples // batch_size))
            i += 1
    except KeyboardInterrupt:
        i += 1
        print("\nWaiting for command, enter exit to terminate, enter help to see all commands")
        while True:
            inp = input()
            if inp == "exit":
                return
            elif inp == "back":
                break
            elif inp == "save":
                saver.save(sess, 'checkpoints/model.ckpt', global_step=i // (num_examples // batch_size))
                return
            elif inp == "test":
                test()
                print("Down! Enter next command, or back to continue training.")
            elif inp == "set batch" or inp == "set test batch":
                print("Enter new batch size.")
                next_inp = input()
                try:
                    if inp == "set batch":
                        b_size = int(next_inp)
                    else:
                        tb_size = int(next_inp)
                except ValueError:
                    print("Failed to set, invalid input")
            elif inp == "help":
                print("=============================")
                print("exit: terminate training")
                print("back: back to training")
                print("save: save the current unfinished epoch and terminate training")
                print("test: print current cost and output test result to /test")
                print("set batch/set test batch")
                print("=============================")

        train(i, b_size, tb_size)


if __name__ == "__main__":
    tagger = Tagger(**attr)
    learning_rate = tf.Variable(starter_learning_rate, trainable=False)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(tagger.cost)

    mnist = data_overlapped.read_data_sets("MNIST_data", n_labeled=num_labeled, one_hot=True)
    saver = tf.train.Saver()

    print("===  Starting Session ===")
    sess = tf.Session()
    i_iter = 0
    ckpt = tf.train.get_checkpoint_state('checkpoints/')  # get latest checkpoint (if any)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        epoch_n = ckpt.model_checkpoint_path.split('-')[1]
        i_iter = (int(epoch_n) + 1) * num_examples // batch_size
        print("Restored Epoch ", epoch_n)
    else:
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        init = tf.global_variables_initializer()
        sess.run(init)
    train(i_iter, batch_size, test_batch_size)
    print("Training finished, testing..")
    test()


