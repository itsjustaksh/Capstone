import os
import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy
import io

print('\n\n\n=============================================')
raw_dataset = tf.data.TFRecordDataset("./TFrecords/train/car-plate.tfrecord")

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    # print(example)

# Create a description of the features.
feature_description = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/bbox/xmax': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'image/bbox/xmin': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'image/bbox/ymax': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'image/bbox/ymin': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    # 'image/object/class/label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    # 'image/object/class/text': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),

}


def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)


parsed_dataset = raw_dataset.map(_parse_function)

# print(parsed_dataset)
# print('done')

# for raw_record in raw_dataset.take(1):
#     example = tf.train.Example()
#     example.ParseFromString(raw_record.numpy())
#     cv2.imshow('test', example)
#     cv2.waitKey(0)

print('Display Monet image...')
for e in parsed_dataset.take(1):
    image = Image.open(io.BytesIO(e['image/encoded'].numpy()))
    image.show()
