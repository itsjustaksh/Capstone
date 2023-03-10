import json
import tensorflow as tf

metaFile = 'tusimple_train_miou=0.2836.ckpt-8.meta'
ckpt = 'model/tusimple/bisenetv2_lanenet/{}'

sess = tf.Session()
saver = tf.train.latest_checkpoint(ckpt)