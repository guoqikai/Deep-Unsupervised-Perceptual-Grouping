import tensorflow as tf
import math


# ======================================== Model functions ===============================================

def corr_input_gauss(inputs, std):
    return inputs + tf.random_normal(tf.shape(inputs)) * std


# the paper suggest we better add bitflip noise when dealing with binary input
def corr_input_bitflip(inputs, std):
    pass


def init_weight(shape, name, seed):
    return tf.Variable(tf.random_normal(shape, name=name, seed=seed) / math.sqrt(shape[0]))


def init_bias(inits, size, name):
    return tf.Variable(inits * tf.ones([size]), name=name)


def batch_normalization(batch, mean=None, var=None):
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch, axes=[0])
    return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))


def g_gauss_stable_v(z_c, u, size):
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


def compute_m_lh(x, z, v):
    return


def compute_z_gradient(x, z, m):
    return


def g_gauss(z_c, u, size):
    """gaussian denoising function proposed in the original paper"""

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

    z_est = (z_c - mu) * v + mu
    return z_est


# ============================================= Tagger =====================================================

class Tagger:

    def __init__(self, **kwargs):

        mandatory_keys = ["iterations", "num_groups", "ladder_layer_sizes"]
        for k in mandatory_keys:
            assert k in kwargs, "missing key " + k

        self.iterations = kwargs.get("iterations")
        self.num_groups = kwargs.get("num_groups")
        self.ladder_layer_sizes = kwargs.get("ladder_layer_sizes")
        self.init_z_val = kwargs.get("init_z_val")

        self.noise_sid = kwargs.get ("noise_sid", 0.3)
        self.seed = kwargs.get("seed", 42)
        self.num_epochs = kwargs.get("num_epochs", 100)
        self.batch_size = kwargs.get("batch_size", 100)

        self.inputs_labeled = tf.placeholder(tf.float32, shape=(None, None))
        self.targets_labeled = tf.placeholder(tf.float32, shape=(None, None, None))
        self.inputs_unlabeled = tf.placeholder(tf.float32, shape=(None, None))

        self.ladder = _TagLadder(self.ladder_layer_sizes, self.seed)

        self._build()


    def _build(self, calculate_ami=False):
        """this method should be called after all fields are inited"""

        clean_out = {}
        corr_out = {}

        if calculate_ami:
            clean_out["l"] = self._build_one_path(self.inputs_labeled, self.targets_labeled, None)
            clean_out["u"] = self._build_one_path(self.inputs_unlabeled, None, None)

        corr_out["l"] = self._build_one_path(self.inputs_labeled, self.targets_labeled, None)
        corr_out["u"] = self._build_one_path (self.inputs_unlabeled, None, lambda x : g_gauss_stable_v(x, self.noise_sid))



    def _build_one_path(self, inputs, labeled_target, noise_func):
        batch_size = tf.shape(inputs)[0]
        input_entries = tf.reduce_prod(tf.shape(inputs)[1:], name="layer_to_first")

        # dim: group, batch, input..
        corr_inputs = tf.keras.backend.repeat_elements(tf.expand_dims(inputs, 0), rep=self.iterations, axis=0)


        if noise_func:
            corr_inputs = noise_func(inputs)

        v_est = tf.Variable(tf.ones[[tf.shape(corr_inputs)]], name="estimated variance of inputs")
        v_est_pos = tf.keras.backend.switch(tf.less(v_est, 0), 1/tf.sqrt(v_est ** 2 + 1), v_est + tf.sqrt(v_est ** 2 + 1))

        attr = {}

        m, z, m_hat, z_hat = (None,) * 4
        for step in range(self.iterations):

            # init/compute the m, z, L(m), delta z mentioned in the paper
            if step == 0:
                m = tf.nn.softmax(tf.random_normal(tf.shape(corr_inputs), seed=self.seed))
                z = tf.fill(tf.shape(corr_inputs), self.init_z_val)
            else:
                m = m_hat
                z = z_hat

            m_lh = compute_m_lh(corr_inputs, z, v_est_pos)
            z_delta = compute_z_gradient(corr_inputs, z, m)

            input = tf.concat([z, z_delta, m, m_lh], axis=2)

            #project the input to ladder net
            self.proj_weight_to_ladder = tf.Variable((input_entries * 4, self.ladder_layer_sizes[0]), name="input project")
            self.proj_bias_to_ladder = tf.Variable((input_entries * 4, self.ladder_layer_sizes[0]), name="input project")



















    def _compute_denoising_cost(self, z_hat, m_hat, x, v):
        """for continuous case Ci = -log(sum(q(x|gk) = N(x; zk, vI))"""







# ======================================== Ladder network ==================================================
# modified version of https://github.com/rinuboney/ladder/blob/master/ladder.py
# should only be used for the tagger framework since it's not a very general implementation of the ladder network
# i.e the decoder only takes inputs from corr encoder


class _TagLadder:

    def __init__(self, layer_sizes, seed):

        self.layer_sizes = layer_sizes
        self.num_layers = num_layers = len(layer_sizes) - 1

        self.ewma = tf.train.ExponentialMovingAverage (decay=0.99)  # to calculate the moving averages of mean and variance
        self.running_mean = [tf.Variable (tf.constant (0.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]
        self.running_var = [tf.Variable (tf.constant (1.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]

        self.weights = \
            {'W': [init_weight(s, "W", seed) for s in zip (layer_sizes[:-1], layer_sizes[1:])],
             'V': [init_weight(s[::-1], "V", seed) for s in zip (layer_sizes[:-1], layer_sizes[1:])],
             'beta': [init_bias(0.0, layer_sizes[l + 1], "beta") for l in range (num_layers)],
             'gamma': [init_bias(1.0, layer_sizes[l + 1], "gamma") for l in range (num_layers)]}

        self.training = tf.placeholder(tf.bool)

    def encoder_path(self, inputs, noise_std, layer_sizes):
        h = inputs + tf.random_normal (tf.shape (inputs)) * noise_std  # add noise to input

        # to store the pre-activation, activation, mean and variance for each layer
        d = {'z': {0: h}, 'm': {}, 'v': {}, 'h': {}}

        for l in range(1, self.num_layers + 1):
            print ("Layer ", l, ": ", layer_sizes[l - 1], " -> ", layer_sizes[l])
            d['h'][l - 1] = h
            z_pre = tf.matmul (h, self.weights['W'][l - 1])  # pre-activation
            m, v = tf.nn.moments(z_pre, axes=[0])

            # if training:
            def training_batch_norm():
                return batch_normalization (z_pre, m, v)

            # else:
            def eval_batch_norm():
                # Evaluation batch normalization
                # obtain average mean and variance and use it to normalize the batch
                mean = self.ewma.average(self.running_mean[l - 1])
                var = self.ewma.average(self.running_var[l - 1])
                z = batch_normalization(z_pre, mean, var)
                return z

            # perform batch normalization according to value of boolean "training" placeholder:
            z = tf.cond(self.training, training_batch_norm, eval_batch_norm)

            if l == self.num_layers:
                # use softmax activation in output layer
                h = tf.nn.softmax(self.weights['gamma'][l - 1] * (z + self.weights["beta"][l - 1]))
            else:
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + self.weights["beta"][l - 1])
            d['z'][l] = z
            d['m'][l], d['v'][l] = m, v
        return h, d

    def decoder_path(self, encoder_attr, corr_top, g_gauss):
        z_est = {}
        for l in range(self.num_layers, -1, -1):
            print("Layer ", l, ": ", self.layer_sizes[l + 1] if l + 1 < len (self.layer_sizes) else None, " -> ",
                  self.layer_sizes[l])
            z, z_c = encoder_attr['z'][l], encoder_attr['z'][l]
            if l == self.num_layers:
                u = corr_top
            else:
                u = tf.matmul(z_est[l + 1], self.weights['V'][l])
            u = batch_normalization(u)
            z_est[l] = g_gauss(z_c, u, self.layer_sizes[l])
        return z_est

