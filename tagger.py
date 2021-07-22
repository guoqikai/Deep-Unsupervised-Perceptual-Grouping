import tensorflow as tf
import math


# ======================================== Model functions ===============================================

def corr_input_gauss(inputs, std):
    return inputs + tf.random_normal(tf.shape(inputs)) * std


def init_weight(shape, name="w"):
    return tf.Variable(tf.random_normal(shape) / math.sqrt(shape[0]), name)


def init_bias(inits, size, name="b"):
    return tf.Variable(inits * tf.ones(size), name)


def batch_normalization(batch, mean=None, var=None, axes=(0,)):
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch, axes=axes)
    return (batch - mean) / tf.sqrt(var + 1e-10)


def compute_m_lh(x, z, v, noise_std):
    noise_factor = noise_std ** 2 + v ** 2
    # Represents negative log-p
    loss = .5 * tf.log(noise_factor + 1e-10) + (z - x) ** 2 / (2. * noise_factor)
    loss -= tf.reduce_min(loss, axis=0, keepdims=True)
    normalizer = tf.log(tf.reduce_sum(tf.exp(-loss), axis=0, keepdims=True) + 1e-10)
    loss += normalizer
    return tf.exp(-loss)


def compute_denoising_cost(z_hat, m_hat, x, v):
    """for continuous case Ci = -log(sum(q(x|gk) = N(x; zk, vI))"""
    sqr_err = (x - z_hat) ** 2
    sqr_err /= v
    log_ps = -0.5 * (tf.log(v + 1e-10) + sqr_err)
    log_ps += tf.math.log(m_hat + + 1e-10)
    log_max = tf.reduce_max(log_ps, axis=0, keepdims=True)
    ps = tf.exp(log_ps - log_max)
    p = tf.reduce_sum(ps, axis=0, keepdims=True)
    err = -tf.log(p + 1e-10) - log_max
    return tf.reduce_mean(err)


def compute_z_gradient(x, z, m):
    return m * (x - z)


def compute_classification_cost(predict, target):
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(target, tf.reduce_sum(predict, axis=0)))


# ============================================= Tagger =====================================================


class Tagger:

    def __init__(self, **kwargs):
        print("=== Init Tagger ===")
        mandatory_keys = ["iterations", "num_groups", "ladder_layer_sizes", "init_z_val", "input_size"]
        for k in mandatory_keys:
            assert k in kwargs, "missing key " + k


        self.iterations = kwargs.get("iterations")
        self.num_groups = kwargs.get("num_groups")
        self.ladder_layer_sizes = kwargs.get("ladder_layer_sizes")
        self.init_z_val = kwargs.get("init_z_val")
        self.input_size = kwargs.get("input_size")

        self.noise_std = kwargs.get("noise_std", 0.1)
        self.class_cost = kwargs.get("class_cost", 0.)

        self.inputs_labeled = tf.placeholder(tf.float32, shape=(None, self.input_size))
        self.targets_labeled = tf.placeholder(tf.float32, shape=(None, None))
        self.inputs_unlabeled = tf.placeholder(tf.float32, shape=(None, self.input_size))
        self.vars = {}

        print("Initiating Ladder..")
        self.ladder = _TagLadder(self.ladder_layer_sizes)
        print("Building graph...")
        self._init_weights()
        self._build()
        print("build success!")

    def _init_weights(self):
        # input variance
        v_raw = tf.Variable(1., "v")
        self.v = tf.keras.backend.switch(tf.less(v_raw, 0), 1. / (tf.sqrt(v_raw ** 2 + 1.) - v_raw),
                                        v_raw + tf.sqrt(v_raw ** 2 + 1.))

        # projection from input to ladder net
        self.l_w = init_weight((self.input_size * 4, self.ladder_layer_sizes[0]), name="l_w")
        self.l_b = init_bias(0., self.ladder_layer_sizes[0], name="l_b")

        # projection from decoder to m_hat and z_hat
        self.w_zh = init_weight((self.ladder_layer_sizes[0], self.input_size), name="w_zh")
        self.b_zh = init_bias(0.,  self.input_size, name="b_zh")

        self.w_mh = init_weight((self.ladder_layer_sizes[0], self.input_size), name="w_mh")

        self.c1 = tf.Variable(tf.ones(self.input_size), name="c1")
        self.c2 = tf.Variable(1., name="c2")

    def _build(self):
        """this method should be called after all fields are inited"""
        unsupervised_attr = self._build_one_path(self.inputs_unlabeled, lambda x: corr_input_gauss(x, self.noise_std))
        self.cost = tf.reduce_mean([compute_denoising_cost(unsupervised_attr["z"][i],
                                                           unsupervised_attr["m"][i],
                                                           self.inputs_unlabeled,
                                                           self.v)
                                    for i in range(self.iterations)])

        self.m = unsupervised_attr["m"]
        self.z = unsupervised_attr["z"]
        self.pred = unsupervised_attr["pred"]

        if self.class_cost > 0:
            supervised_attr = self._build_one_path(self.inputs_labeled, lambda x: corr_input_gauss(x, self.noise_std))
            self.cost += compute_classification_cost(supervised_attr["pred"][-1], self.targets_labeled) * self.class_cost
            self.m = tf.concat([self.m, supervised_attr["m"]], axis=2)
            self.z = tf.concat([self.z, supervised_attr["z"]], axis=2)
            self.pred = tf.concat([self.pred, supervised_attr["pred"]], axis=2)

    def _build_one_path(self, inputs, noise_func):
        inputs = tf.expand_dims(inputs, 0)
        corr_inputs = inputs
        if noise_func:
            corr_inputs = noise_func(inputs)

        # dim: group, batch, input..
        corr_inputs = tf.keras.backend.repeat_elements(corr_inputs, rep=self.num_groups, axis=0)

        attr = {"pred": [], "m": [], "z": []}

        m, z, m_hat, z_hat = (None,) * 4
        for step in range(self.iterations):

            # init/compute the m, z, L(m), delta z mentioned in the paper
            if step == 0:
                m = tf.nn.softmax(tf.random_normal(tf.shape(corr_inputs)), axis=0)
                z = tf.fill(tf.shape(corr_inputs), self.init_z_val)
            else:
                m = m_hat
                z = z_hat

            m_lh = compute_m_lh(corr_inputs, z, self.v, self.noise_std)
            z_delta = compute_z_gradient(corr_inputs, z, m)

            inputs_to_ladder = tf.concat([z, z_delta, m, m_lh], axis=2)

            input_shape = tf.shape(inputs_to_ladder)
            # flatten group
            inputs_to_ladder = tf.reshape(inputs_to_ladder, [-1, input_shape[2]])

            # create one hidden layer
            h = tf.nn.relu(batch_normalization(tf.matmul(inputs_to_ladder, self.l_w)) + self.l_b)

            # pass to ladder
            top, encoder_attr = self.ladder.encoder_path(h)

            # get z_est of the lowest layer
            z_est = self.ladder.decoder_path(encoder_attr, top)[0]

            z_hat = self.c1 * tf.reshape((batch_normalization(tf.matmul(z_est, self.w_zh)) +
                                          self.b_zh), (input_shape[0], input_shape[1], self.input_size))

            m_hat = tf.nn.softmax(self.c2 * tf.reshape(batch_normalization(tf.matmul(z_est, self.w_mh)),
                                                       (input_shape[0], input_shape[1], self.input_size)), axis=0)

            top_reshaped = tf.reshape(top, shape=(input_shape[0], input_shape[1], -1))
            pred = top_reshaped / tf.reduce_sum(tf.reduce_sum(top_reshaped, axis=2, keepdims=True), axis=0, keepdims=True)

            attr["pred"].append(pred)
            attr["m"].append(m_hat)
            attr["z"].append(z_hat)

        return attr


# ======================================== Ladder network ==================================================
# modified version of https://github.com/rinuboney/ladder/blob/master/ladder.py
# should only be used for the tagger framework since it"s not a very general implementation of the ladder network
# i.e the decoder only takes inputs from corr encoder


class _TagLadder:

    def __init__(self, layer_sizes):

        self.layer_sizes = layer_sizes
        self.num_layers = num_layers = len(layer_sizes) - 1

        self.ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
        self.running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]
        self.running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]

        self.weights = \
            {"w": [init_weight(s) for s in zip(layer_sizes[:-1], layer_sizes[1:])],
             "v": [init_weight(s[::-1], "v") for s in zip(layer_sizes[:-1], layer_sizes[1:])],
             "beta": [init_bias(0.0, layer_sizes[l + 1], "beta") for l in range(num_layers)],
             "gamma": [init_bias(1.0, layer_sizes[l + 1], "gamma") for l in range(num_layers)]}

        self.g_weights = {}

        for l in range(self.num_layers, -1, -1):
            wi = lambda inits, name: tf.Variable(inits * tf.ones([self.layer_sizes[l]]), name=name)
            self.g_weights[l] = {}
            self.g_weights[l]["a1"] = wi(0, "a1")
            self.g_weights[l]["a2"] = wi(1., "a2")
            self.g_weights[l]["a3"] = wi(0., "a3")
            self.g_weights[l]["a4"] = wi(0., "a4")
            self.g_weights[l]["a5"] = wi(0., "a5")
            self.g_weights[l]["a6"] = wi(0., "a6")
            self.g_weights[l]["a7"] = wi(1., "a7")
            self.g_weights[l]["a8"] = wi(0., "a8")
            self.g_weights[l]["a9"] = wi(0., "a9")
            self.g_weights[l]["a10"] = wi(0., "a10")

    def encoder_path(self, inputs):
        h = inputs

        # to store the pre-activation, activation, mean and variance for each layer
        d = {"z": {0: h}, "m": {}, "v": {}, "h": {}}

        for l in range(1, self.num_layers + 1):
            d["h"][l - 1] = h
            z_pre = tf.matmul(h, self.weights["w"][l - 1])  # pre-activation
            m, v = tf.nn.moments(z_pre, axes=[0])
            z = batch_normalization(z_pre, m, v)

            if l == self.num_layers:
                # use softmax activation in output layer
                h = tf.nn.softmax(self.weights["gamma"][l - 1] * (z + self.weights["beta"][l - 1]))
            else:
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + self.weights["beta"][l - 1])
            d["z"][l] = z
            d["m"][l], d["v"][l] = m, v
        return h, d

    def g_gauss(self, z_c, u, l):
        """for continuous case, put v through a sigmoid"""
        gv = lambda name: self.g_weights[l][name]
        mu = gv("a1") * tf.sigmoid(gv("a2") * u + gv("a3")) + gv("a4") * u + gv("a5")
        v = gv("a6") * tf.sigmoid(gv("a7") * u + gv("a8")) + gv("a9") * u + gv("a10")
        v = tf.sigmoid(v)

        z_est = (z_c - mu) * v + mu
        return z_est

    def decoder_path(self, encoder_attr, corr_top):
        z_est = {}
        for l in range(self.num_layers, -1, -1):
            z_c = encoder_attr["z"][l]
            if l == self.num_layers:
                u = corr_top
            else:
                u = tf.matmul(z_est[l + 1], self.weights["v"][l])
            u = batch_normalization(u)
            z_est[l] = self.g_gauss(z_c, u, l)
        return z_est

