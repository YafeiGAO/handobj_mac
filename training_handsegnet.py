#
#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function, unicode_literals

import tensorflow as tf
import os
import sys
import numpy as np

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from data.BinaryDbReader import BinaryDbReader
from utils.general import LearningRateScheduler, load_weights_from_snapshot
from libs.dataset_utils import create_input_pipeline
from libs.vae import VAE
import argparse
#from pca import pca
from libs import utils
parser = argparse.ArgumentParser(description='Parser added')
parser.add_argument(
        '-c',
        action="store_true",
        dest="convolutional", help='Whether use convolution or not')
parser.add_argument(
        '-f',
        action="store_true",
        dest="fire", help='Whether use fire module or not')
parser.add_argument(
        '-v',
        action="store_true",
        dest="variational", help='Wether use latent variance or not')
parser.add_argument(
        '-m',
        action="store_true",
        dest="metric", help='Whether use metric loss or not')
parser.add_argument(
        '-r',
        action="store",
        type=int,
        dest="rank", help='Rank of metric learning')
parser.add_argument(
        '-o',
        action="store",
        dest="output_path",
        default="result_vae", help='Destination for storing results')
parser.print_help()
results = parser.parse_args()
print(results)
learning_rate=0.0001
n_files_train = 11020
input_shape=[320, 320, 3]
output_shape=[320, 320,1]
crop_shape=[256, 256]
n_filters=[256, 128, 128, 128, 128, 128]
n_hidden=128
n_code=64
n_clusters=4
dropout=True
filter_sizes=[3, 3, 3, 3, 3, 3]
keep_prob = 0.8
ckpt_name="vae.ckpt"
n_examples=6
batch_size = 8
# training parameters
train_para = {'lr': [1e-5, 1e-6, 1e-7],
              'lr_iter': [20000, 30000],
              'max_iter': 40000,
              'show_loss_freq': 1000,
              'snapshot_freq': 5000,
              'snapshot_dir': 'snapshots_handsegnet_variational'}

# get dataset
dataset = BinaryDbReader(mode='training',
                         batch_size=batch_size, shuffle=True,
                         hue_aug=True, random_crop_to_size=True)

# build network graph
data = dataset.get()
print(data)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# build network
evaluation = tf.placeholder_with_default(True, shape=())
#net = ColorHandPose3DNetwork()

ae = VAE(input_shape=[None] + crop_shape + [input_shape[-1]],
         output_shape=[None] + crop_shape + [output_shape[-1]],
         convolutional=results.convolutional,
         fire=results.fire,
         variational=results.variational,
         metric=results.metric,
         order=results.rank,
         n_filters=n_filters,
         n_hidden=n_hidden,
         n_code=n_code,
         n_clusters=n_clusters,
         dropout=dropout,
         filter_sizes=filter_sizes,
         activation=tf.nn.sigmoid)

np.random.seed(1)
# print(np.random.get_state())
zs = np.random.uniform(
    -1.0, 1.0, [4, n_code]).astype(np.float32)
zs = utils.make_latent_manifold(zs, 6)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(ae['cost'])
#train_op = optimizer.minimize(loss)

# config.gpu_options.per_process_gpu_memory_fraction = 0.4

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
output_path=results.output_path
train_writer = tf.summary.FileWriter(output_path + '/logs', sess.graph)
print(output_path + '/logs')
cost = 0

coord = tf.train.Coordinator()
tf.get_default_graph().finalize()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
if (
        os.path.exists(output_path + '/' + ckpt_name + '.index') or
        os.path.exists(ckpt_name)
    ):
    saver.restore(sess, output_path + '/' + ckpt_name)
    print("Model restored")

train_data = sess.run(data)
#print(train_data)
train_xs = train_data['image']
train_ts = train_data['hand_parts']
train_ys = train_data['label']


for idx in range(0, 4):
    temp_data = sess.run(data)
    temp_xs = temp_data['image']

    #utils.montage(temp_xs[:2**2],output_path + '/input_train_%s.png'%str(idx))
    temp_ts = temp_data['hand_parts']

    #utils.montage(temp_ts[:2**2],output_path + '/target_train_%s.png'%str(idx))
    temp_ys = temp_data['label']

    train_xs = np.append(train_xs, temp_xs, axis=0)
    train_ts = np.append(train_ts, temp_ts, axis=0)
    train_ys = np.append(train_ys, temp_ys, axis=0)


utils.montage(train_xs[:n_examples**2],output_path + '/input_train.png')
utils.montage(train_ts[:n_examples**2],output_path + '/target_train.png')


# snapshot dir
if not os.path.exists(train_para['snapshot_dir']):
    os.mkdir(train_para['snapshot_dir'])
    print('Created snapshot dir:', train_para['snapshot_dir'])

# Training loop
batch_i = 0
for i in range(train_para['max_iter']):
    batch_data = sess.run(data)
    batch_xs = batch_data['image']
    batch_ts = batch_data['hand_parts']
    batch_ys = batch_data['label']

    #print(batch_ts.shape)

    batch_ts = np.expand_dims(batch_ts, axis=3)
    train_cost, _ = sess.run([ae['cost'], optimizer], feed_dict={ae['x']: batch_xs, ae['t']: batch_ts, ae['label']: batch_ys[:, 0], ae['train']: True, ae['keep_prob']: keep_prob})
    # write summary
    train_ts = np.reshape(train_ts, (-1, 256, 256, 1))
    summary = sess.run(ae['merged'], feed_dict={ae['x']: train_xs,ae['t']: train_ts, ae['label']: train_ys[:, 0], ae['train']: False, ae['keep_prob']: 1.0})
    nebula3d = sess.run(ae['nebula3d'], feed_dict={ae['x']: batch_xs, ae['train']: False, ae['keep_prob']: 1.0})
    cost += train_cost
    print('Iter %d\t Loss %.1e' % (i, train_cost))
    '''
    batch_i += 1
    if batch_i % (n_files_train//batch_size) == 0:
        train_writer.add_summary(summary,i*(n_files_train//batch_size) + batch_i)
        print('epoch:', i)
        print('training cost:', cost / batch_i)
        cost = 0
        batch_i = 0
    '''

    if (i % train_para['show_loss_freq']) == 0:
        print('Iteration %d\t Loss %.1e' % (i, cost))
        recon = sess.run(ae['y'], feed_dict={ae['z']: zs,ae['train']: False,ae['keep_prob']: 1.0})
        utils.montage(recon.reshape([-1] + crop_shape + [output_shape[-1]]),output_path + '/manifold.png')
        for cat in range(nebula3d.shape[0]):
            recon = sess.run(ae['y'], feed_dict={ae['z']: zs/1.2+nebula3d[cat, :],ae['train']: False,ae['keep_prob']: 1.0})
            utils.montage(recon.reshape([-1] + crop_shape + [output_shape[-1]]),output_path + '/manifold_%03d.png' % cat)

        recon = sess.run(ae['y'], feed_dict={ae['x']: train_xs[:n_examples**2],ae['train']: False,ae['keep_prob']: 1.0})
        utils.montage(recon.reshape([-1] + crop_shape + [output_shape[-1]]),output_path+'/recon_train.png')
        sys.stdout.flush()

    if (i % train_para['snapshot_freq']) == 0:
        saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=i)
        print('Saved a snapshot.')
        sys.stdout.flush()


print('Training finished. Saving final snapshot.')
saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=train_para['max_iter'])
