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
import os
import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d, load_weights_from_snapshot
USE_RETRAINED = True
PATH_TO_POSENET_SNAPSHOTS = './snapshots_posenet_0620_only/'  # only used when USE_RETRAINED is true
PATH_TO_HANDSEGNET_SNAPSHOTS = './vae_0626+vm/'  # only used when USE_RETRAINED is true
PATH_TO_LIFTING_SNAPSHOTS = './objall/snapshots_lifting_proposed_objall/'
ckpt_name="vae.ckpt"
if __name__ == '__main__':
    # images to be shown
    image_list = list()

    image_list.append('/home/gaoyafei/hand3d/1.jpeg')
    image_list.append('/home/gaoyafei/original/data/img.png')
    image_list.append('/home/gaoyafei/original/data/img2.png')
    image_list.append('/home/gaoyafei/original/data/img3.png')
    image_list.append('/home/gaoyafei/original/data/img4.png')
    image_list.append('/home/gaoyafei/original/data/img5.png')
    image_list.append('/home/gaoyafei/original/data/image1.jpeg')
    image_list.append('/home/gaoyafei/original/data/image2.jpeg')
    image_list.append('/home/gaoyafei/original/data/image3.jpeg')
    image_list.append('/home/gaoyafei/original/data/image4.jpeg')
    image_list.append('./test/00_00011.png')
    image_list.append('./test/00_00012.png')
    image_list.append('./test/00_00013.png')
    image_list.append('./test/00_00014.png')
    image_list.append('./test/00_00015.png')
    image_list.append('./test/00_00016.png')
    image_list.append('./test/00_00017.png')
    

    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, 256, 256, 3))
    target_tf = tf.placeholder(tf.float32, shape=(1, 256, 256, 1))
    hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
    evaluation = tf.placeholder_with_default(True, shape=())
    #label = tf.zeros((1,1))
    #ppshots_posenet_0611_onlyrint(label)
    # build work
    net = ColorHandPose3DNetwork()
    print('net')
    #print(image_tf)
    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
    keypoints_scoremap_tf, keypoint_coord3d_tf,cost_handseg_tf = net.inference(image_tf, target_tf,hand_side_tf, evaluation)

    # Start TF
    config = tf.ConfigProto()
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=4.0)
    tf.train.start_queue_runners(sess=sess)
    #coord = tf.train.Coordinator()
    #tf.get_default_graph().finalize()
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # initialize network weights
    #import ipdb;ipdb.set_trace()
    #sess.run(tf.global_variables_initializer())
    if USE_RETRAINED:
        # retrained version: HandSegNet
        '''
        if (os.path.exists(PATH_TO_HANDSEGNET_SNAPSHOTS + ckpt_name + '.index') or                            
                os.path.exists(ckpt_name)):
            #sess1 = tf.Session(config=config)
            #saver1 = tf.train.Saver()
            #sess1.run(tf.global_variables_initializer())
            saver.restore(sess, PATH_TO_HANDSEGNET_SNAPSHOTS + ckpt_name)
            print('HandSeg Model restored')
        
        if (os.path.exists(PATH_TO_POSENET_SNAPSHOTS + '/' + 'model-80000.index') or
                os.path.exists( 'model-80000')):
            saver.restore(PATH_TO_POSENET_SNAPSHOTS + '/' + 'model-80000')
            print("Pose Model restored")
        
        if (os.path.exists(PATH_TO_LIFTING_SNAPSHOTS + '/' + 'model-80000.index') or
                os.path.exists( 'model-80000')):
            saver.restore(sess, PATH_TO_LIFTING_SNAPSHOTS + '/' + 'model-80000')
            print("Lifting Model restored")
        '''
        last_cpt = tf.train.latest_checkpoint(PATH_TO_HANDSEGNET_SNAPSHOTS)
        assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
        load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'beta'])
        
        # retrained version: PoseNet
        last_cpt = tf.train.latest_checkpoint(PATH_TO_POSENET_SNAPSHOTS)
        assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
        load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])

        # retrained version: HandSegNet
        last_cpt = tf.train.latest_checkpoint(PATH_TO_LIFTING_SNAPSHOTS)
        assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network?"
        load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
    else:
        # load weights used in the paper
        net.init(sess)

    ig_name = 0
    # Feed image list through network
    for img_name in image_list:
        ig_name += 1
        image_raw = scipy.misc.imread(img_name)

        image_raw = image_raw[:,:,0:3]

        image_raw = scipy.misc.imresize(image_raw, (256, 256))
        #image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)
        image_v = np.expand_dims((image_raw.astype('float') / 255.0) , 0)
        target_v = image_v[:,:,:,0]
        target_v = np.expand_dims((target_v) , 3)
        print(target_v.shape)
        print(image_v.shape)
        print('image_v in run.py:')
        #print(image_v)
        #import ipdb;ipdb.set_trace()
        feed_dict = {image_tf: image_v, target_tf:target_v}
        #import ipdb;ipdb.set_trace()
        hand_scoremap_v, image_crop_v, scale_v, center_v,keypoints_scoremap_v, keypoint_coord3d_v,cost_handseg_v = sess.run(
                [hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
                    keypoints_scoremap_tf, keypoint_coord3d_tf,cost_handseg_tf], feed_dict=feed_dict)

        hand_scoremap_v = np.squeeze(hand_scoremap_v)
        image_crop_v = np.squeeze(image_crop_v)
        keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
        keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

        # post processing
        #image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
        image_crop_v = ((image_crop_v ) * 255).astype('uint8')
        coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
        coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

        # visualize
        
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224, projection='3d')
        ax1.imshow(image_raw)
        plot_hand(coord_hw, ax1)
        ax2.imshow(image_crop_v)
        plot_hand(coord_hw_crop, ax2)
        ax3.imshow(np.argmax(hand_scoremap_v, 2))
        plot_hand_3d(keypoint_coord3d_v, ax4)
        ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
        ax4.set_xlim([-3, 3])
        ax4.set_ylim([-3, 1])
        ax4.set_zlim([-3, 3])
        #plt.show()
        '''
        ax1 = plt.subplot(221)
        plt.imshow(image_raw)
        plot_hand(coord_hw, ax1)
        ax2 = plt.subplot(222)
        plt.imshow(image_crop_v)
        plot_hand(coord_hw_crop, ax2)
        ax3 = plt.subplot(223)
        plt.imshow(np.argmax(hand_scoremap_v,2))
        ax4 = plt.subplot(224,projection='3d')
        plot_hand_3d(keypoint_coord3d_v,ax4)
        ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
        ax4.set_xlim([-3, 3])
        ax4.set_ylim([-3, 1])
        ax4.set_zlim([-3, 3])
        '''
        plt.savefig('./result_0628/%s.png'%ig_name,format='png', dpi=300)
