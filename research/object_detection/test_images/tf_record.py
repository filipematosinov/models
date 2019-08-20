import tensorflow as tf
import re
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import sys
sys.path.append("../../")
from object_detection.utils import dataset_util
sys.path.remove("../../")

flags = tf.app.flags
flags.DEFINE_string('output_path', 'my_images/tf_record_dataset', 'Path to output TFRecord')
FLAGS = flags.FLAGS

IMAGES = "my_images/frames/"
LABELS = "my_images/labels/"

LABEL_DICT = {
  "person": 1
}


def create_tf_example(example):
    # TODO(user): Populate the following variables from your example.

    frame_label = read_frame_label(example[:-4])

    if frame_label.empty:
        return 0

    height = frame_label['height'].values[0]  # Image height
    width = frame_label['width'].values[0]  # Image width
    filename = example.encode()  # Filename of the image. Empty if image is not from file
    with tf.gfile.GFile(IMAGES+example, 'rb') as fid:
        encoded_image_data = fid.read()

    image_format = 'jpeg'.encode()  # b'jpeg' or b'png'
    xmins = (frame_label['x1']/width).tolist()
    # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = (frame_label['x2']/width).tolist()  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = (frame_label['y1']/height).tolist()
    # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = (frame_label['y2']/height).tolist()  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    #classes_text = frame_label.apply(lambda x: x.encode(), columnns=['class']).tolist()
    classes_text = frame_label['class'].apply(lambda x: x.encode()).tolist()  # List of string class name of bounding box (1 per box)
    classes = np.ones(len(frame_label), dtype=int).tolist()  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(
        feature={
            'image/height':
            dataset_util.int64_feature(height),
            'image/width':
            dataset_util.int64_feature(width),
            'image/filename':
            dataset_util.bytes_feature(filename),
            'image/source_id':
            dataset_util.bytes_feature(filename),
            'image/encoded':
            dataset_util.bytes_feature(encoded_image_data),
            'image/format':
            dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymaxs),
            'image/object/class/text':
            dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label':
            dataset_util.int64_list_feature(classes),
        }))
    return tf_example


def read_frame_label(frame_name):  # frame_name should be without the jpg extension
    # Extract the sequence name & open the csv
    num = re.findall('\d+', frame_name)
    video_id = num[0]
    cam_id = num[1]
    seq_id = num[2]
    frame_id = int(num[3])
    seq_name = "%s_cam_%s_%s.csv" % (video_id, cam_id, seq_id)
    frame_name = "%s_cam_%s_%s_%d" % (video_id, cam_id, seq_id, frame_id)
    try:
        frame_label = pd.read_csv(LABELS + seq_name)
    except:
        return pd.DataFrame()
    frame_label = frame_label.loc[frame_label['frame'] == frame_name]

    return frame_label


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # TODO(user): Write code to read in your dataset to examples variable
    # Reading out dataset

    examples = [f for f in listdir(IMAGES) if isfile(join(IMAGES, f))]

    counter = 0
    len_examples = len(examples)
    for example in examples:
        tf_example = create_tf_example(example)
        if tf_example != 0:
            writer.write(tf_example.SerializeToString())

        if counter % 100 == 0:
            print("Percent done", (counter / len_examples) * 100)
        counter += 1

    writer.close()


if __name__ == '__main__':
    tf.app.run()
