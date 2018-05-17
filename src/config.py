import tensorflow as tf

flags = tf.app.flags

first_index = ord(' ')
last_index = ord('~')

flags.DEFINE_string('img_path', '../data/test_images', 'Path location of images for training')
flags.DEFINE_string('log_path', '../data/log', 'Log location for model stats')
flags.DEFINE_string('save_path', '../data/sess', 'Save location for model session')
flags.DEFINE_integer('seed', 7, 'Random seed')
flags.DEFINE_boolean('remove_log', True, 'Remove log on start')
flags.DEFINE_integer('height', 32, 'Images height')
flags.DEFINE_integer('n_channels', 3, 'Images channels')
flags.DEFINE_integer('first_index', first_index, 'First index of encoded character')
flags.DEFINE_integer('last_index', last_index, 'Last index of encoded character')
flags.DEFINE_integer('n_classes', last_index - first_index, 'Number of classes of the prediction')
flags.DEFINE_float('learning_rate', 0.005, 'Learning rate')
flags.DEFINE_integer('batch_size', 32, 'Number of samples for each training epoch')
flags.DEFINE_integer('n_filters', 12, 'Number of filters for each CNN')
flags.DEFINE_integer('n_rnn_cells', 2, 'Number of cells rnn layer')
flags.DEFINE_integer('n_units', 256, 'Number of units in each rnn cell')
flags.DEFINE_integer('n_epochs', 100000, 'Number of training epoch')
flags.DEFINE_float('keep_prob', 0.7, 'Dropout for rnn')
flags.DEFINE_float('train_test_ratio', 0.7, 'Sampling ratio for train/test split')

FLAGS = flags.FLAGS
