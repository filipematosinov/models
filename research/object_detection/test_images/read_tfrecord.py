import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.enable_eager_execution()
data_path = 'results/detections.tfrecord'  # address to save the hdf5 file

raw_image_dataset = tf.data.TFRecordDataset(data_path)

# Create a dictionary describing the features.
def read_tfrecord(serialized_example):

    feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature((), tf.int64)

    }
    example = tf.io.parse_single_example(serialized_example, feature_description)

    image = tf.io.parse_tensor(example['image/encoded'], out_type=float)
    image_shape = [example['image/height'], example['image/width'], 3]
    image = tf.reshape(image, image_shape)
    return image, example['image/object/class/label']


parsed_image_dataset, _ = raw_image_dataset.map(read_tfrecord)

for data in parsed_image_dataset:

  img = tf.keras.preprocessing.image.array_to_img(data[0])

  plt.imshow(img)

  plt.show()
