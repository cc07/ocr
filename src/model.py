import tensorflow as tf
import numpy as np

class CtcOcr:

    def __init__(self,
                 n_features,
                 n_channels,
                 n_classes,
                 n_filters=64,
                 n_rnn_cells=1,
                 n_units=512,
                 learning_rate=0.005,
                 keep_prob=1,
                 save_path='../data/sess/sess.ckpt',
                 log_path='../data/log',
                 seed=7,
                 sess=None,
                 ):

        self.n_features = n_features
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_filters = n_filters
        self.n_rnn_cells = n_rnn_cells
        self.n_units = n_units
        self.keep_prob = keep_prob

        self.save_path = save_path
        self.log_path = log_path

        self.graph = tf.Graph()

        with self.graph.as_default() as graph:
            tf.set_random_seed(seed)

            self._build_optimizer()
            self._build_net()
            self.sess = sess if not sess is None else tf.Session(graph=graph)
            self.saver = tf.train.Saver()

            self.summary_writer = {}

            for stage in ['train', 'test', 'valid']:
                self.summary_writer[stage] = tf.summary.FileWriter('{}/{}'.format(self.log_path, stage), graph)

            self.sess.run(tf.global_variables_initializer())

    def _default_w_init(self, seed=7):
        return tf.contrib.layers.variance_scaling_initializer(seed=seed)

    def _default_b_init(self):
        return tf.constant_initializer(0.1)

    def _default_conv_layer(self, inputs, n_filters, kernel_size, w_init, b_init, activation=tf.nn.elu, padding='VALID'):
        conv = tf.layers.conv2d(
            inputs,
            n_filters,
            kernel_size=kernel_size,
            kernel_initializer=w_init,
            bias_initializer=b_init,
            activation=None,
            padding=padding,
        )

        conv = tf.contrib.layers.batch_norm(conv)

        if not activation is None:
            conv = activation(conv)

        return conv

    def _default_max_pool(self, inputs, pool_size=(2, 2), strides=(2, 2)):
        return tf.layers.max_pooling2d(
            inputs,
            pool_size=pool_size,
            strides=strides,
        )

    def _default_avg_pool(self, inputs, pool_size=(3, 3), strides=(2, 2)):
        return tf.layers.average_pooling2d(
            inputs,
            pool_size=pool_size,
            strides=strides,
        )

    def _build_res_layer(self, inputs, n_filters, w_init, b_init):
        with tf.variable_scope('conv1'):
            conv1 = self._default_conv_layer(inputs, n_filters, (3, 3), w_init, b_init, padding='SAME')

        with tf.variable_scope('conv2'):
            conv2 = self._default_conv_layer(conv1, n_filters, (3, 3), w_init, b_init, padding='SAME', activation=None)

        residual = tf.contrib.layers.batch_norm(conv2)
        residual = tf.nn.elu(residual)

        return residual + inputs

    def _build_conv_layer(self, inputs):
        w_init = self._default_w_init()
        b_init = self._default_b_init()

        with tf.variable_scope('stem'):
            stem = self._default_conv_layer(inputs, self.n_filters, (3, 3), w_init, b_init)
            stem = self._default_conv_layer(stem, self.n_filters, (3, 3), w_init, b_init)
            stem = self._default_max_pool(stem)

        with tf.variable_scope('residual1'):
            residual = stem

            for i in range(5):
                with tf.variable_scope('res_1_{}'.format(i)):
                    residual = self._build_res_layer(residual, self.n_filters, w_init, b_init)
            residual = self._default_max_pool(residual)

        with tf.variable_scope('residual2'):
            for i in range(10):
                with tf.variable_scope('res_2_{}'.format(i)):
                    residual = self._build_res_layer(residual, self.n_filters, w_init, b_init)
            residual = self._default_max_pool(residual)

        with tf.variable_scope('residual3'):
            for i in range(5):
                with tf.variable_scope('res_3_{}'.format(i)):
                    residual = self._build_res_layer(residual, self.n_filters, w_init, b_init)
            residual = self._default_max_pool(residual)

        with tf.variable_scope('conv'):
            conv = self._default_conv_layer(residual, self.n_filters * 2, (1, 1), w_init, b_init)
            conv = self._default_conv_layer(conv, self.n_filters * 4, (1, 1), w_init, b_init)
            conv = self._default_max_pool(conv, (3, 1), (1, 1))
            conv = self._default_conv_layer(conv, self.n_filters * 8, (1, 1), w_init, b_init)

        return conv

    def _build_rnn_cells(self, init):
        cells = []
        for _ in range(self.n_rnn_cells):
            cell = tf.contrib.rnn.LSTMCell(self.n_units, initializer=init)
            cells.append(cell)

        return tf.contrib.rnn.MultiRNNCell(cells)

    def _build_rnn_layer(self, inputs, seq_len, seed=7):
        init = tf.orthogonal_initializer(seed=seed)

        with tf.variable_scope('rnn_fw'):
            cell_fw = self._build_rnn_cells(init)

        with tf.variable_scope('rnn_bw'):
            cell_bw = self._build_rnn_cells(init)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, seq_len, dtype=tf.float32)
        outputs = tf.concat(outputs, 2)

        return outputs

    def _build_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

    def _compute_train_cost(self, logits, seq_len):
        return tf.reduce_mean(tf.nn.ctc_loss(labels=self.targets,
                                             inputs=logits,
                                             sequence_length=seq_len,
                                             ))
    def _build_summary(self):
        cost = tf.summary.scalar('cost', self.cost)
        ler = tf.summary.scalar('ler', self.ler)

        self.merged_summary = tf.summary.merge([cost, ler])

    def _build_net(self):
        self.inputs = tf.placeholder(tf.float32, [None, None, self.n_features, self.n_channels])
        self.targets = tf.sparse_placeholder(tf.int32)

        self.global_step = tf.Variable(0, trainable=False)

        shape = tf.shape(self.inputs)
        conv = self._build_conv_layer(self.inputs)

        _, feature_w, feature_h, feature_c = conv.get_shape().as_list()

        conv = tf.reshape(conv, [shape[0], -1, feature_h * feature_c])
        seq_len = tf.fill([shape[0]], tf.shape(conv)[1])

        rnn_outputs = self._build_rnn_layer(conv, seq_len)

        outputs = tf.reshape(rnn_outputs, [-1, self.n_units * 2])

        W = tf.Variable(tf.truncated_normal([self.n_units * 2, self.n_classes], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[self.n_classes]))

        logits = tf.matmul(outputs, W) + b
        logits = tf.reshape(logits, [shape[0], -1, self.n_classes])
        logits = tf.transpose(logits, (1, 0, 2))

        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

        self.dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
        self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), self.targets))

        self.cost = self._compute_train_cost(logits, seq_len)
        self.cost += 1 - self.ler
        self.train_op = self.optimizer.minimize(self.cost, global_step=self.global_step)

        self._build_summary()

    def fit(self, targets, inputs):
        cost, ler, summary, global_step, _ = self.sess.run([
                                self.cost,
                                self.ler,
                                self.merged_summary,
                                self.global_step,
                                self.train_op,
                             ],
                             feed_dict={
                                self.targets: targets,
                                self.inputs: inputs,
                             })

        self.summary_writer['train'].add_summary(summary, global_step)

        return cost, ler

    def validate(self, targets, inputs, stage='test'):
        cost, ler, summary, global_step = self.sess.run([
                                self.cost,
                                self.ler,
                                self.merged_summary,
                                self.global_step,
                             ],
                             feed_dict={
                                self.targets: targets,
                                self.inputs: inputs,
                             })

        self.summary_writer[stage].add_summary(summary, global_step)

        return cost, ler

    def predict(self, image):
        return self.sess.run(
                            self.dense_decoded,
                            feed_dict={
                                self.inputs: image[np.newaxis, :],
                            })

    def save(self, path=None):
        save_path = path if path is not None else self.save_path

        self.saver.save(self.sess, save_path, global_step=self.global_step)
        print('Saving sess to {}: {}'.format(save_path))

    def load(self, checkpoint_dir='../data/sess'):
        try:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        except:
            ckpt = None

        if not (ckpt and ckpt.model_checkpoint_path):
            print('Cannot find any saved sess in checkpoint_dir')
        else:
            try:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)

                print('Sess restored successfully: {}'.format(ckpt.model_checkpoint_path))
            except Exception as e:
                print('Failed to load sess: {}'.format(str(e)))

if __name__ == '__main__':
    model = CtcOcr(32, 3, 15)
