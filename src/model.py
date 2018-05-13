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
            self._build_optimizer()
            self._build_net()
            self.sess = sess if not sess is None else tf.Session(graph=graph)
            self.saver = tf.train.Saver()

            self.summary_writer = {}

            for stage in ['train', 'test', 'valid']:
                self.summary_writer[stage] = tf.summary.FileWriter('{}/{}'.format(self.log_path, stage), graph)

            self.sess.run(tf.global_variables_initializer())

    def _default_w_init(self):
        return tf.contrib.layers.variance_scaling_initializer()

    def _default_b_init(self):
        return tf.constant_initializer(0.1)

    def _default_conv_layer(self, inputs, n_filters, kernel_size, activation=tf.nn.elu):
        return tf.layers.conv2d(
            inputs,
            n_filters,
            kernel_size=kernel_size,
            padding='SAME',
            kernel_initializer=self._default_w_init(),
            bias_initializer=self._default_b_init(),
            activation=activation,
        )

    def _default_max_pool(self, inputs, pool_size=(2, 2), strides=(2, 2)):
        return tf.layers.max_pooling2d(
            inputs,
            pool_size=pool_size,
            strides=strides,
            padding='SAME',
        )

    def _default_avg_pool(self, inputs):
        return tf.layers.average_pooling2d(
            inputs,
            pool_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
        )

    def _build_inception_a_layer(self, inputs, n_filters):
        with tf.variable_scope('conv1'):
            conv1 = self._default_conv_layer(inputs, n_filters, (1, 1))

        with tf.variable_scope('conv2'):
            conv2 = self._default_conv_layer(inputs, n_filters, (1, 1))
            conv2 = self._default_conv_layer(conv2, n_filters, (1, 3))
            conv2 = self._default_conv_layer(conv2, n_filters, (3, 1))

        with tf.variable_scope('conv3'):
            conv3 = self._default_conv_layer(inputs, n_filters, (1, 1))
            conv3 = self._default_conv_layer(conv3, n_filters, (1, 5))
            conv3 = self._default_conv_layer(conv3, n_filters, (5, 1))
            conv3 = self._default_conv_layer(conv3, n_filters, (1, 5))
            conv3 = self._default_conv_layer(conv3, n_filters, (5, 1))

        with tf.variable_scope('conv4'):
            conv4 = self._default_avg_pool(inputs)
            conv4 = self._default_conv_layer(conv4, n_filters, (1, 1))

        with tf.variable_scope('concat'):
            inception = tf.concat([conv1, conv2, conv3, conv4], axis=3)
            inception = self._default_conv_layer(inception, n_filters * 4, (1, 1), activation=None)
            inception = inception + inputs

        return inception

    def _build_reduction_a_layer(self, inputs, n_filters):
        with tf.variable_scope('conv1'):
            conv1 = self._default_conv_layer(inputs, n_filters * 3, (3, 3))

        with tf.variable_scope('conv2'):
            conv2 = self._default_conv_layer(inputs, n_filters, (1, 1))
            conv2 = self._default_conv_layer(conv2, n_filters, (3, 3))
            conv2 = self._default_conv_layer(conv2, n_filters * 2, (3, 3))

        with tf.variable_scope('conv3'):
            conv3 = self._default_max_pool(inputs, pool_size=(3, 3), strides=(1, 1))

        with tf.variable_scope('concat'):
            reduction = tf.concat([conv1, conv2, conv3], axis=3)

        return reduction

    def _build_inception_b_layer(self, inputs, n_filters):
        with tf.variable_scope('conv1'):
            conv1 = self._default_conv_layer(inputs, n_filters * 2, (1, 1))

        with tf.variable_scope('conv2'):
            conv2 = self._default_conv_layer(inputs, n_filters * 2, (1, 1))
            conv2 = self._default_conv_layer(conv2, n_filters * 2, (1, 7))
            conv2 = self._default_conv_layer(conv2, n_filters * 2, (7, 1))

        with tf.variable_scope('concat'):
            inception = tf.concat([conv1, conv2], axis=3)
            inception = self._default_conv_layer(inception, n_filters * 9, (1, 1), activation=None)
            inception = inception + inputs

        return inception

    def _build_reduction_b_layer(self, inputs, n_filters):
        with tf.variable_scope('conv1'):
            conv1 = self._default_conv_layer(inputs, n_filters * 4, (1, 1))
            conv1 = self._default_conv_layer(conv1, n_filters * 6, (3, 3))

        with tf.variable_scope('conv2'):
            conv2 = self._default_conv_layer(inputs, n_filters * 4, (1, 1))
            conv2 = self._default_conv_layer(conv2, n_filters * 4, (3, 3))
            conv2 = self._default_conv_layer(conv2, n_filters * 4, (3, 3))

        with tf.variable_scope('conv3'):
            conv3 = self._default_max_pool(inputs, pool_size=(3, 3), strides=(1, 1))

        with tf.variable_scope('conv4'):
            conv4 = self._default_conv_layer(inputs, n_filters * 4, (1, 1))
            conv4 = self._default_conv_layer(conv4, n_filters * 6, (3, 3))

        with tf.variable_scope('concat'):
            reduction = tf.concat([conv1, conv2, conv3, conv4], axis=3)
            reduction = self._default_conv_layer(reduction, n_filters * 8, (1, 1), activation=None)

        return reduction

    def _build_inception_c_layer(self, inputs, n_filters):
        with tf.variable_scope('conv1'):
            conv1 = self._default_conv_layer(inputs, n_filters * 4, (1, 1))

        with tf.variable_scope('conv2'):
            conv2 = self._default_conv_layer(inputs, n_filters * 4, (1, 1))
            conv2 = self._default_conv_layer(conv2, n_filters * 4, (1, 3))
            conv2 = self._default_conv_layer(conv2, n_filters * 4, (3, 1))

        with tf.variable_scope('concat'):
            inception = tf.concat([conv1, conv2], axis=3)
            inception = self._default_conv_layer(inception, n_filters * 8, (1, 1), activation=None)
            inception = inception + inputs

        return inception

    def _build_stem(self, inputs, n_filters):
        stem = self._default_conv_layer(inputs, n_filters, (3, 3))
        stem = self._default_max_pool(stem)
        stem = self._default_conv_layer(stem, n_filters * 2, (3, 3))
        stem = self._default_max_pool(stem)
        stem = self._default_conv_layer(stem, n_filters * 2, (1, 1))
        stem = self._default_conv_layer(stem, n_filters * 3, (3, 3))
        stem = self._default_conv_layer(stem, n_filters * 4, (3, 3))
        stem = self._default_max_pool(stem)

        return stem

    def _build_conv_layer(self, inputs):

        with tf.variable_scope('stem'):
            stem = self._build_stem(inputs, self.n_filters)

        with tf.variable_scope('inception_a'):
            inception_a = stem

            for i in range(5):
                with tf.variable_scope('inception_a_{}'.format(i)):
                    inception_a = self._build_inception_a_layer(inception_a, self.n_filters)

        with tf.variable_scope('reduction_a'):
            reduction_a = self._build_reduction_a_layer(inception_a, self.n_filters)

        with tf.variable_scope('inception_b'):
            inception_b = reduction_a

            for i in range(10):
                with tf.variable_scope('inception_b_{}'.format(i)):
                    inception_b = self._build_inception_b_layer(inception_b, self.n_filters)

        with tf.variable_scope('reduction_b'):
            reduction_b = self._build_reduction_b_layer(inception_b, self.n_filters)

        with tf.variable_scope('reduction_c'):
            inception_c = reduction_b

            for i in range(5):
                with tf.variable_scope('inception_c_{}'.format(i)):
                    inception_c = self._build_inception_c_layer(inception_c, self.n_filters)

        with tf.variable_scope('pooling'):
            pooling = self._default_avg_pool(inception_c)

        return pooling

    def _build_rnn_cells(self):
        cells = []
        for _ in range(self.n_rnn_cells):
            cell = tf.contrib.rnn.LSTMCell(self.n_units, initializer=tf.orthogonal_initializer())
            cells.append(cell)

        return tf.contrib.rnn.MultiRNNCell(cells)

    def _build_rnn_layer(self, inputs, seq_len):
        with tf.variable_scope('rnn_fw'):
            cells_fw = self._build_rnn_cells()

        with tf.variable_scope('rnn_bw'):
            cells_bw = self._build_rnn_cells()

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs, seq_len, dtype=tf.float32)
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
        seq_len = tf.fill([shape[0]], shape[1] // 8)

        rnn_outputs = self._build_rnn_layer(conv, seq_len)

        outputs = tf.reshape(rnn_outputs, [-1, self.n_units * 2])

        W = tf.Variable(tf.truncated_normal([self.n_units * 2, self.n_classes], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[self.n_classes]))

        logits = tf.matmul(outputs, W) + b
        logits = tf.reshape(logits, [shape[0], -1, self.n_classes])
        logits = tf.transpose(logits, (1, 0, 2))

        self.cost = self._compute_train_cost(logits, seq_len)
        self.train_op = self.optimizer.minimize(self.cost, global_step=self.global_step)

        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

        self.dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
        self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), self.targets))

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
