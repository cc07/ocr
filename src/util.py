import cv2
import numpy as np
import tensorflow as tf

from config import FLAGS

def normalize(layer):
    return layer / 255. - 0.5

def denormalize(layer):
    return layer * 255 + 0.5

def restore_image(normalized_img):
    img = np.transpose(normalized_img, (1, 2, 0))
    img = [denormalize(layer) for layer in img]

    return np.transpose(img, (2, 0, 1))

def resize_image(image, new_h):
    ratio = float(image.shape[1]) / float(image.shape[0])
    new_w = int((new_h * ratio) + 0.5)

    return cv2.resize(image, (new_w, new_h))

def create_train_batch(batch_size, labels, images, is_training=True, dtype=np.int64):
    indices = []
    values = []

    sample_index = np.random.choice(len(images) - 1, size=batch_size)
    sample_images = []

    x_max = 0
    y_max = 0

    for key, index in zip(sample_index, range(batch_size)):
        sample = labels[key]
        length = len(sample)
        label = sample

        img = images[key]
        x_max = max(x_max, img.shape[0])
        y_max = max(y_max, img.shape[1])

        # if is_training:
        #     img = salt_pepper(img)

        sample_images.append(img)

        for number, number_index in zip(label, range(length)):
            indices.append((index, number_index))
            values.append(ord(str(number)) - FLAGS.first_index)

    indices = np.array(indices, dtype=np.int64)
    values = np.array(values, dtype=dtype)
    shape = np.array([batch_size, np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    targets = tf.SparseTensorValue(indices=indices, values=values, dense_shape=shape)
    sample_images = add_padding_to_images(x_max, y_max, sample_images)

    print('Images shape: {}'.format(sample_images[0].shape))

    return targets, sample_images

def add_padding_to_images(x_max, y_max, images):

    output = []
    for img in images:
        left_pad = int((x_max - img.shape[0]) / 2)
        right_pad = x_max - img.shape[0] - int((x_max - img.shape[0]) / 2)
        top_pad = int((y_max - img.shape[1]) / 2)
        bottom_pad = y_max - img.shape[1] - int((y_max - img.shape[1]) / 2)

        img = np.pad(img,
                     pad_width=((left_pad, right_pad),
                                (top_pad, bottom_pad),
                                (0, 0)),
                     mode='constant',
                     constant_values=0)

        img = np.transpose(img.reshape((x_max, y_max, FLAGS.n_channels)), [1, 0, 2])
        output.append(img)

    return np.array(output)

def salt_pepper(image):
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0

    return out

def train_test_split(labels, images, train_test_ratio):
    n_images = len(images)
    n_test_index = int(n_images * train_test_ratio)
    n_valid_index = (n_images - n_test_index) // 2

    train_images = images[:n_test_index]
    test_images = images[n_test_index:n_test_index + n_valid_index]
    valid_images = images[n_test_index + n_valid_index:]

    train_labels = labels[:n_test_index]
    test_labels = labels[n_test_index:n_test_index + n_valid_index]
    valid_labels = labels[n_test_index + n_valid_index:]

    print('train size: {}, test size: {}, valid size: {}'.format(len(train_images), len(test_images), len(valid_images)))

    return train_labels, train_images, test_labels, test_images, valid_labels, valid_images
