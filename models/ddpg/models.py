import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, h1, h2, model='mlp', name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.h1 = h1
        self.h2 = h2

        assert model in ['mlp', 'cnn']
        if model == 'mlp':
            self.base = MLP(layer_norm=layer_norm,
                            h1=h1,
                            h2=None,
                            out_dim=h1,
                            scope=self.name + '_base',
                            act_fn='relu')
        else:
            self.base = CNN(layer_norm=layer_norm,
                            h1=h1,
                            h2=h2,
                            out_dim=h1,
                            scope=self.name + '_base',
                            act_fn='relu')

        self.head = MLP(layer_norm=layer_norm,
                        h1=h2,
                        h2=None,
                        out_dim=nb_actions,
                        scope=self.name + '_head',
                        act_fn='tanh')

    def __call__(self, obs, reuse=False):
        '''
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, self.h1)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.h2)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.nb_actions,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                 maxval=3e-3))
            x = tf.nn.tanh(x)
        return x
        '''
        x = self.base(obs, reuse=reuse)
        x = self.head(x, reuse=reuse)

        return x


class Critic(Model):
    def __init__(self, h1, h2, model='mlp', name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)

        self.layer_norm = layer_norm
        self.h1 = h1
        self.h2 = h2

        assert model in ['mlp', 'cnn']
        if model == 'mlp':
            self.obs_proc = MLP(layer_norm=layer_norm,
                                h1=h1,
                                h2=None,
                                out_dim=h1,
                                scope=self.name + '_base',
                                act_fn='relu')
        else:
            self.obs_proc = CNN(layer_norm=layer_norm,
                                h1=h1,
                                h2=h2,
                                out_dim=h1,
                                scope=self.name + '_base',
                                act_fn='relu')

        self.head = MLP(layer_norm=layer_norm,
                        h1=h2,
                        h2=None,
                        out_dim=1,
                        scope=self.name + '_head',
                        act_fn='linear')

    def __call__(self, obs, action, reuse=False):
        '''
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, self.h1)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            print('Action shape: {}'.format(action.shape))
            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, self.h2)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                 maxval=3e-3))
        return x
        '''
        x = self.obs_proc(obs, reuse=reuse)
        x = tf.concat([x, action], axis=-1)
        x = self.head(x, reuse=reuse)

        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars


class MLP(object):

    def __init__(self, layer_norm, h1, h2, out_dim, scope, act_fn='linear'):
        self.scope = scope

        assert isinstance(layer_norm, bool)
        self.layer_norm = layer_norm

        assert (isinstance(h1, int) or h1 is None)
        self.h1 = h1

        assert (isinstance(h2, int) or h2 is None)
        self.h2 = h2

        assert isinstance(out_dim, int)
        self.out_dim = out_dim

        assert act_fn in ['linear', 'tanh', 'relu']
        self.act_fn = act_fn

    def __call__(self, inp, reuse=False):
        with tf.variable_scope(self.scope) as scope:
            if reuse:
                scope.reuse_variables()

            x = inp
            x = tf.layers.flatten(x)
            if self.h1 is not None:
                x = tf.layers.dense(x, self.h1)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

            if self.h2 is not None:
                x = tf.layers.dense(x, self.h2)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.out_dim,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                 maxval=3e-3))
            if self.act_fn == 'tanh':
                x = tf.nn.tanh(x)
            elif self.act_fn == 'relu':
                x = tf.nn.relu(x)

        return x


class CNN(object):
    ''' Stack of 2 convolutional layers ending in flattened layer.
        h1 and h2 are number of filters rather than layer widths.
    '''
    def __init__(self, layer_norm, h1, h2, out_dim, scope, act_fn='linear'):
        self.scope = scope

        assert isinstance(layer_norm, bool)
        self.layer_norm = layer_norm

        assert (isinstance(h1, int) or h1 is None)
        self.h1 = h1

        assert (isinstance(h2, int) or h2 is None)
        self.h2 = h2

        assert isinstance(out_dim, int)
        self.out_dim = out_dim

        assert act_fn in ['linear', 'tanh', 'relu']
        self.act_fn = act_fn

    def __call__(self, inp, reuse=False):
        print('reuse: {}'.format(reuse))
        with tf.variable_scope(self.scope) as scope:
            if reuse:
                scope.reuse_variables()

            x = tf.reshape(inp, [-1] + list(inp.shape[2:4]) + [inp.shape[1] * inp.shape[4]])
            if self.h1 is not None:
                x = tf.layers.conv2d(x, self.h1, kernel_size=3)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

            if self.h2 is not None:
                x = tf.layers.conv2d(x, self.h2, kernel_size=3)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, self.out_dim,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                 maxval=3e-3))
            if self.act_fn == 'tanh':
                x = tf.nn.tanh(x)
            elif self.act_fn == 'relu':
                x = tf.nn.relu(x)

        return x
