import os
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from PIL import Image
import imageio
from glob import glob
from typing import List


def get(name):
    return tf.compat.v1.get_default_graph().get_tensor_by_name(name + ':0')


def tensorflow_session():
    # Init session and params
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    # Pin GPU to local rank (one GPU per process)
    config.gpu_options.visible_device_list = str(0)
    sess = tf.compat.v1.Session(config=config)
    return sess


def load_model_and_params():
    graph_path = 'models/graph_unoptimized.pb'
    inputs = {
        'dec_eps_0': 'Placeholder',
        'dec_eps_1': 'Placeholder_1',
        'dec_eps_2': 'Placeholder_2',
        'dec_eps_3': 'Placeholder_3',
        'dec_eps_4': 'Placeholder_4',
        'dec_eps_5': 'Placeholder_5',
        'enc_x': 'input/image',
        'enc_x_d': 'input/downsampled_image',
        'enc_y': 'input/label'
    }
    outputs = {
        'dec_x': 'model_1/Cast_1',
        'enc_eps_0': 'model/pool0/truediv_1',
        'enc_eps_1': 'model/pool1/truediv_1',
        'enc_eps_2': 'model/pool2/truediv_1',
        'enc_eps_3': 'model/pool3/truediv_1',
        'enc_eps_4': 'model/pool4/truediv_1',
        'enc_eps_5': 'model/truediv_4'
    }

    with tf.io.gfile.GFile(graph_path, 'rb') as f:
        graph_def_optimized = tf.compat.v1.GraphDef()
        graph_def_optimized.ParseFromString(f.read())

    sess = tensorflow_session()
    tf.import_graph_def(graph_def_optimized)
    n_eps = 6

    # Encoder
    enc_x = get(inputs['enc_x'])
    enc_eps = [get(outputs['enc_eps_' + str(i)]) for i in range(n_eps)]
    enc_x_d = get(inputs['enc_x_d'])
    enc_y = get(inputs['enc_y'])

    # Decoder
    dec_x = get(outputs['dec_x'])
    dec_eps = [get(inputs['dec_eps_' + str(i)]) for i in range(n_eps)]

    eps_shapes = [(128, 128, 6), (64, 64, 12), (32, 32, 24),
                  (16, 16, 48), (8, 8, 96), (4, 4, 384)]
    eps_sizes = [np.prod(e) for e in eps_shapes]
    eps_size = 256 * 256 * 3
    res = edict({'sess': sess,
                 'enc_x': enc_x,
                 'enc_eps': enc_eps,
                 'enc_x_d': enc_x_d,
                 'n_eps': n_eps,
                 'enc_y': enc_y,
                 'dec_x': dec_x,
                 'dec_eps': dec_eps,
                 'eps_shapes': eps_shapes,
                 'eps_sizes': eps_sizes,
                 'eps_size': eps_size})
    return res


def update_feed(feed_dict, bs, enc_x_d, enc_y):
    x_d = 128 * np.ones([bs, 128, 128, 3], dtype=np.uint8)
    y = np.zeros([bs], dtype=np.int32)
    feed_dict[enc_x_d] = x_d
    feed_dict[enc_y] = y
    return feed_dict

def run(sess, fetches, feed_dict, lock):
    with lock:
        # Locked tensorflow so average server response time to user is lower
        result = sess.run(fetches, feed_dict)
    return result


def flatten_eps(eps):
    # [BS, eps_size]
    return np.concatenate([np.reshape(e, (e.shape[0], -1)) for e in eps], axis=-1)


def unflatten_eps(feps, eps_shapes):
    index = 0
    eps = []
    bs = feps.shape[0]  # feps.size // eps_size
    for shape in eps_shapes:
        eps.append(np.reshape(
            feps[:, index: index+np.prod(shape)], (bs, *shape)))
        index += np.prod(shape)
    return eps


def encode(img, enc_x, sess, enc_eps, enc_x_d, enc_y, lock):
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    bs = img.shape[0]
    assert img.shape[1:] == (256, 256, 3)
    feed_dict = {enc_x: img}

    update_feed(feed_dict, bs, enc_x_d, enc_y)  # For unoptimized model
    return flatten_eps(run(sess, enc_eps, feed_dict, lock))


def decode(feps, dec_x, sess, dec_eps, eps_shapes, n_eps, enc_x_d, enc_y, lock, **kwargs):
    if len(feps.shape) == 1:
        feps = np.expand_dims(feps, 0)
    bs = feps.shape[0]
    eps = unflatten_eps(feps, eps_shapes)

    feed_dict = {}
    for i in range(n_eps):
        feed_dict[dec_eps[i]] = eps[i]

    update_feed(feed_dict, bs, enc_x_d, enc_y)  # For unoptimized model
    return run(sess, dec_x, feed_dict, lock)


def random(params_dict, bs=1, eps_std=0.7, eps_size=256 * 256 * 3):
    feps = np.random.normal(scale=eps_std, size=[bs, eps_size])
    return decode(feps, **params_dict), feps