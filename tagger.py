import tensorflow as tf
import math


# ======================================== Model functions ===============================================

def corr_input_gauss(inputs, std):
    return inputs + tf.random_normal(tf.shape(inputs)) * std


def init_weight(shape, name, seed):
    return tf.Variable(tf.random_normal(shape, name=name, seed=seed) / math.sqrt(shape[0]))


def init_bias(inits, size, name):
    return tf.Variable(inits * tf.ones(size), name=name)


def batch_normalization(batch, mean=None, var=None):
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch, axes=[0])
    return (batch - mean) / tf.sqrt(var + 1e-10)


def g_gauss(z_c, u, size):
    "for continuous case, put v through a sigmoid"
    wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
    a1 = wi(0., 'a1')
    a2 = wi(1., 'a2')
    a3 = wi(0., 'a3')
    a4 = wi(0., 'a4')
    a5 = wi(0., 'a5')

    a6 = wi(0., 'a6')
    a7 = wi(1., 'a7')
    a8 = wi(0., 'a8')
    a9 = wi(0., 'a9')
    a10 = wi(0., 'a10')

    mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
    v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10
    v = tf.sigmoid(v)

    z_est = (z_c - mu) * v + mu
    return z_est


def compute_m_lh(x, z, v, noise_std):
    noise_factor = noise_std ** 2 + v ** 2
    # Represents negative log-p
    loss = 0.5 * tf.log(noise_factor) + tf.sqrt(z - x) / (2. * noise_factor)
    loss -= tf.reduce_min(loss, axis=0, keepdims=True)
    normalizer = tf.log(tf.reduce_sum(tf.exp(-loss), axis=0, keepdims=True))
    loss += normalizer
    return tf.exp(-loss)


def compute_z_gradient(x, z, m):
    return m * (x - z)


def compute_classification_cost(predict, target):
    return 0


def compute_denoising_cost(z_hat, m_hat, x, v):
    """for continuous case Ci = -log(sum(q(x|gk) = N(x; zk, vI))"""

    input_shape = tf.shape(x)
    flat_shape = (input_shape[0], input_shape[1], -1)
    z_hat = tf.reshape(z_hat, flat_shape)
    m_hat = tf.reshape(m_hat, flat_shape)
    x = tf.reshape(x, flat_shape)
    v = tf.reshape(v, flat_shape)

    sqr_err = (x - z_hat) ** 2
    sqr_err /= v

    log_ps = .5 * (tf.log(v) + sqr_err)
    log_ps += tf.log (m_hat + 1e-5)

    log_max = tf.reduce_max(log_ps, axis=0, keepdims=True)
    ps = tf.exp(log_ps - log_max)

    p = tf.reduce_sum(ps, axis=0, keepdims=True)
    err = -tf.log(p) - log_max

    return tf.reduce_mean(err)

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

        self.noise_std = kwargs.get("noise_std", 0.3)
        self.class_cost = kwargs.get("class_cost", 0.1)
        self.seed = kwargs.get("seed", 42)

        self.inputs_labeled = tf.placeholder(tf.float32, shape=(None, self.input_size))
        self.targets_labeled = tf.placeholder(tf.float32, shape=(None, None))
        self.inputs_unlabeled = tf.placeholder(tf.float32, shape=(None, self.input_size))

        print("Initiating Ladder..")
        self.ladder = _TagLadder(self.ladder_layer_sizes, self.seed)
        print("Building graph...")
        self._build()
        print("build success")

    def _build(self, calculate_ami=False):
        """this method should be called after all fields are inited"""

        clean_out = {}
        corr_out = {}

        if calculate_ami:
            clean_out["l"] = self._build_one_path(self.inputs_labeled, self.targets_labeled, None)
            clean_out["u"] = self._build_one_path(self.inputs_unlabeled, None, None)

        corr_out["l"] = self._build_one_path(self.inputs_labeled, self.targets_labeled, None)
        corr_out["u"] = self._build_one_path(self.inputs_unlabeled, None, lambda x: corr_input_gauss(x, self.noise_std))

        self.cost = tf.reduce_mean(corr_out["u"]["cost"]) + corr_out["l"]["cost"][-1] * self.class_cost
        self.m = corr_out["u"]["m"]
        self.z = corr_out["u"]["z"]

    def _build_one_path(self, inputs, labeled_target, noise_func):

        # dim: group, batch, input..
        corr_inputs = tf.keras.backend.repeat_elements(tf.expand_dims(inputs, 0), rep=self.iterations, axis=0)

        if noise_func:
            corr_inputs = noise_func(corr_inputs)
        v_est = tf.Variable(1.0, name="estimated_variance_inputs") * tf.ones_like(corr_inputs)
        v_est_pos = tf.keras.backend.switch(tf.less(v_est, 0), 1/tf.sqrt(v_est ** 2 + 1), v_est + tf.sqrt(v_est ** 2 + 1))

        attr = {"cost": [], "pred": [], "m": [], "z": []}

        m, z, m_hat, z_hat = (None,) * 4
        for step in range(self.iterations):

            # init/compute the m, z, L(m), delta z mentioned in the paper
            if step == 0:
                m = tf.nn.softmax(tf.random_normal(tf.shape(corr_inputs), seed=self.seed))
                z = tf.fill(tf.shape(corr_inputs), self.init_z_val)
            else:
                m = m_hat
                z = z_hat

            m_lh = compute_m_lh(corr_inputs, z, v_est_pos, self.noise_std)
            z_delta = compute_z_gradient(corr_inputs, z, m)

            inputs = tf.concat([z, z_delta, m, m_lh], axis=2)

            # project the input to ladder net
            proj_weight_to_ladder = \
                init_weight((self.input_size * 4, self.ladder_layer_sizes[0]), "input_project_weight", self.seed)
            proj_bias_to_ladder = \
                init_bias(1., self.ladder_layer_sizes[0], "input_project_bias")

            # flatten group
            input_shape = tf.shape(inputs)
            inputs = tf.reshape(inputs, [-1, input_shape[2]])

            # create one hidden layer
            h = tf.nn.relu(batch_normalization(tf.matmul(inputs, proj_weight_to_ladder)) + proj_bias_to_ladder)


            # pass to ladder
            top, encoder_attr = self.ladder.encoder_path(h)

            # get z_est of the lowest layer
            z_est = self.ladder.decoder_path(encoder_attr, top)[0]


            proj_weight_z_hat = \
                init_weight((self.ladder_layer_sizes[0], self.input_size), "z_hat_project_weight", self.seed)
            proj_bias_z_hat = \
                init_bias(1.,  self.input_size, "z_hat_project_bias")

            z_hat = tf.reshape(tf.Variable(tf.ones(self.input_size), name="c1") *
                               (batch_normalization(tf.matmul(z_est, proj_weight_z_hat)) +
                                proj_bias_z_hat), (-1, input_shape[1], self.input_size))

            proj_weight_m_hat = \
                init_weight((self.ladder_layer_sizes[0], self.input_size), "m_hat_project_weight", self.seed)

            m_hat = tf.reshape(tf.nn.softmax(tf.Variable(1.0, name="c2") *
                                             batch_normalization(tf.matmul(z_est, proj_weight_m_hat)), axis=0),
                               (-1, input_shape[1], self.input_size))

            top_reshaped = tf.reshape(top, shape=(input_shape[0], input_shape[1], -1))
            pred = top_reshaped / tf.reduce_sum(tf.reduce_sum(top_reshaped, axis=2, keepdims=True), axis=0, keepdims=True)

            if labeled_target is not None:
                attr["cost"].append(compute_classification_cost(pred, labeled_target))
            else:
                attr["cost"].append(compute_denoising_cost(z_hat, m_hat, corr_inputs, v_est))

            attr["pred"].append(pred)
            attr["m"].append(m_hat)
            attr["z"].append(z_hat)

        return attr



# ======================================== Ladder network ==================================================
# modified version of https://github.com/rinuboney/ladder/blob/master/ladder.py
# should only be used for the tagger framework since it's not a very general implementation of the ladder network
# i.e the decoder only takes inputs from corr encoder


class _TagLadder:

    def __init__(self, layer_sizes, seed):

        self.layer_sizes = layer_sizes
        self.num_layers = num_layers = len(layer_sizes) - 1

        self.ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
        self.running_mean = [tf.Variable (tf.constant(0.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]
        self.running_var = [tf.Variable (tf.constant(1.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]

        self.weights = \
            {'W': [init_weight(s, "W", seed) for s in zip(layer_sizes[:-1], layer_sizes[1:])],
             'V': [init_weight(s[::-1], "V", seed) for s in zip(layer_sizes[:-1], layer_sizes[1:])],
             'beta': [init_bias(0.0, layer_sizes[l + 1], "beta") for l in range(num_layers)],
             'gamma': [init_bias(1.0, layer_sizes[l + 1], "gamma") for l in range(num_layers)]}


    def encoder_path(self, inputs):
        h = inputs

        # to store the pre-activation, activation, mean and variance for each layer
        d = {'z': {0: h}, 'm': {}, 'v': {}, 'h': {}}

        for l in range(1, self.num_layers + 1):
            d['h'][l - 1] = h
            z_pre = tf.matmul(h, self.weights['W'][l - 1])  # pre-activation
            m, v = tf.nn.moments(z_pre, axes=[0])
            z = batch_normalization (z_pre, m, v)

            if l == self.num_layers:
                # use softmax activation in output layer
                h = tf.nn.softmax(self.weights['gamma'][l - 1] * (z + self.weights["beta"][l - 1]))
            else:
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + self.weights["beta"][l - 1])
            d['z'][l] = z
            d['m'][l], d['v'][l] = m, v
        return h, d

    def decoder_path(self, encoder_attr, corr_top):
        z_est = {}
        for l in range(self.num_layers, -1, -1):
            z, z_c = encoder_attr['z'][l], encoder_attr['z'][l]
            if l == self.num_layers:
                u = corr_top
            else:
                u = tf.matmul(z_est[l + 1], self.weights['V'][l])
            u = batch_normalization(u)
            z_est[l] = g_gauss(z_c, u, self.layer_sizes[l])
        return z_est

