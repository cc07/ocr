import numpy as np
import cv2
import re
import shutil

from os import listdir
from os.path import isfile, join

from model import CtcOcr
from config import FLAGS
from util import normalize, denormalize, preprocess_image, restore_image, create_train_batch, resize_image, train_test_split

def load_data(img_path, height=64):
    labels = []
    images = []
    images_path = []

    print('Loading images...')

    for f in listdir(img_path):
        path = join(img_path, f)

        if isfile(path) and 'jpg' in path:
            img = cv2.imread(path)
            # img = preprocess_image(img, height)
            img = resize_image(img, height)

            label = re.match(r'.*[_](.*)\.jpg', f).group(1)
            # label = label.replace('-', ' ')

            images.append(img)
            labels.append(label)

    print('Loading images...completed, total: {}/{} images'.format(len(images), len(labels)))

    return labels, images

def train(model, n_epochs, labels, images, batch_size, train_test_ratio):

    train_labels, train_images, test_labels, test_images, valid_labels, valid_images = train_test_split(labels, images, train_test_ratio)

    for epoch in range(n_epochs):

        train_targets, sample_images = create_train_batch(batch_size, train_labels, train_images)

        train_cost, train_ler = model.fit(train_targets, sample_images)
        train_ler = train_ler * batch_size

        # test_targets, test_sample_images = create_train_batch(batch_size, test_labels, test_images, False)
        #
        # test_cost, test_ler = model.validate(test_targets, test_sample_images)
        # test_ler = test_ler * batch_size
        #
        # valid_targets, valid_sample_images = create_train_batch(batch_size, valid_labels, valid_images, False)
        #
        # valid_cost, valid_ler = model.validate(valid_targets, valid_sample_images, 'valid')
        # valid_ler = valid_ler * batch_size

        log = '[Epoch {}] train_cost: {:.3f}, train_ler: {:.3f}, test_cost: {:.3f}, test_ler: {:.3f}, val_cost: {:.3f}, val_ler: {:.3f}'
        # print(log.format(epoch, train_cost, train_ler, test_cost, test_ler, valid_cost, valid_ler))
        print(log.format(epoch, train_cost, train_ler, 0, 0, 0, 0))

        if train_cost < 0:
            model.save()

if __name__ == '__main__':

    if FLAGS.remove_log:
        shutil.rmtree(FLAGS.log_path, ignore_errors=True)

    labels, images = load_data(FLAGS.img_path, FLAGS.height)
    model = CtcOcr(FLAGS.height,
                   FLAGS.n_channels,
                   FLAGS.n_classes,
                   FLAGS.n_filters,
                   FLAGS.n_rnn_cells,
                   FLAGS.n_units,
                   FLAGS.learning_rate,
                   FLAGS.keep_prob,
                   FLAGS.save_path,
                   FLAGS.log_path,
                   )

    train(model, FLAGS.n_epochs, labels, images, FLAGS.batch_size, FLAGS.train_test_ratio)
