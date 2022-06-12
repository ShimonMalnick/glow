import os

import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
from threading import Lock
import imageio
from glob import glob
from typing import List

lock = Lock()


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

# optimized = True
optimized = False

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

def update_feed(feed_dict, bs):
    x_d = 128 * np.ones([bs, 128, 128, 3], dtype=np.uint8)
    y = np.zeros([bs], dtype=np.int32)
    feed_dict[enc_x_d] = x_d
    feed_dict[enc_y] = y
    return feed_dict

with tf.io.gfile.GFile(graph_path, 'rb') as f:
    graph_def_optimized = tf.compat.v1.GraphDef()
    graph_def_optimized.ParseFromString(f.read())

sess = tensorflow_session()
tf.import_graph_def(graph_def_optimized)

print("Loaded model")

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
# z_manipulate = np.load('models/z_manipulate.npy')


def run(sess, fetches, feed_dict):
    with lock:
        # Locked tensorflow so average server response time to user is lower
        result = sess.run(fetches, feed_dict)
    return result


def flatten_eps(eps):
    # [BS, eps_size]
    return np.concatenate([np.reshape(e, (e.shape[0], -1)) for e in eps], axis=-1)


def unflatten_eps(feps):
    index = 0
    eps = []
    bs = feps.shape[0]  # feps.size // eps_size
    for shape in eps_shapes:
        eps.append(np.reshape(
            feps[:, index: index+np.prod(shape)], (bs, *shape)))
        index += np.prod(shape)
    return eps


def encode(img):
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    bs = img.shape[0]
    assert img.shape[1:] == (256, 256, 3)
    feed_dict = {enc_x: img}

    update_feed(feed_dict, bs)  # For unoptimized model
    return flatten_eps(run(sess, enc_eps, feed_dict))


def decode(feps):
    if len(feps.shape) == 1:
        feps = np.expand_dims(feps, 0)
    bs = feps.shape[0]
    eps = unflatten_eps(feps)

    feed_dict = {}
    for i in range(n_eps):
        feed_dict[dec_eps[i]] = eps[i]

    update_feed(feed_dict, bs)  # For unoptimized model
    return run(sess, dec_x, feed_dict)


def random(bs=1, eps_std=0.7):
    feps = np.random.normal(scale=eps_std, size=[bs, eps_size])
    return decode(feps), feps


def test():

    img = Image.open('demo/test/img.png')
    img = np.reshape(np.array(img), [1, 256, 256, 3])

    # Encoding speed
    eps = encode(img)
    n_samples = 100
    # std = 0.08
    dists: List[float] = np.linspace(0.05, 0.8, 16).tolist()
    for d in dists:
        os.mkdir(f'samples/{d:.3f}')
        new_eps = np.random.normal(scale=d, size=[n_samples, eps_size]) + eps

        for i in range(n_samples):
            dec = decode(new_eps[i])
            img = Image.fromarray(dec[0])
            img.save(f'samples/{d:.3f}/{i}.png')

    for d in dists:
        paths = glob(f'samples/{d:.3f}/*.png')
        video = imageio.get_writer(f'samples/vids/dist_{d:.3f}.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        for im in paths:
            video.append_data(imageio.imread(im))
        video.close()


def random_images(batch_size=16, n_imgs=64, eps_std=0.7):
    n_iters = n_imgs // batch_size
    images  = []
    for i in range(n_iters):
        tensors, _ = random(batch_size, eps_std)
        images += [Image.fromarray(tensors[i]) for i in range(batch_size)]

    for i in range(len(images)):
        images[i].save('samples/random/im_' + str(i) + '.png')
    paths = glob(f'samples/random/*.png')
    video = imageio.get_writer(f'samples/random/vid.mp4', mode='I', fps=2, codec='libx264', bitrate='16M')
    for im in paths:
        video.append_data(imageio.imread(im))
    video.close()


def interpolate_2_images(im1_path, im2_path, alpha_steps=41, alpha_max=2.0):
    steps = np.linspace(0, alpha_max, alpha_steps)
    im1, im2 = Image.open(im1_path), Image.open(im2_path)
    im1 = np.reshape(np.array(im1), [1, 256, 256, 3])
    im2 = np.reshape(np.array(im2), [1, 256, 256, 3])
    z_1 = encode(im1)
    z_2 = encode(im2)
    delta_z = z_2 - z_1
    with open(f"experiments/interpolation/info.txt", "w") as f:
        f.write(f"images paths: im1: {im1_path}\nim2: {im2_path}\n")
        f.write(f'alpha_steps: {alpha_steps}\nalpha_max: {alpha_max}\n')
    for i in range(alpha_steps):
        cur_z = z_1 + delta_z * steps[i]
        cur_image = decode(cur_z)
        Image.fromarray(cur_image[0]).save(f'experiments/interpolation/step_{i}.png')


def create_video_with_labels(out_path, paths, labels, fps=2, codec='libx264', bitrate='16M'):
    video = imageio.get_writer(out_path, mode='I', fps=fps, codec=codec, bitrate=bitrate)
    images = [Image.open(path) for path in paths]
    font = ImageFont.truetype('f1.ttf', 16)
    for i in range(len(paths)):
        cur_draw = ImageDraw.Draw(images[i])
        cur_draw.text((150, 0), labels[i], font=font, fill=(255, 0, 0))
        video.append_data(np.asarray(images[i]))

    video.close()

# warm start
_img, _z = random(1)
_z = encode(_img)
print("Warm started tf model")

if __name__ == '__main__':
    # interpolate_2_images('experiments/interpolation/im1.png', 'experiments/interpolation/im2.png')
    labels: List[float] = np.linspace(0, 2, 41).tolist()
    labels: List[str] = ['alpha = ' + str(round(l, 2)) for l in labels]
    paths = [f'experiments/interpolation/images/step_{i}.png' for i in range(len(labels))]
    create_video_with_labels('experiments/interpolation/vid.mp4', paths, labels)
    # random_images()
    # test()
