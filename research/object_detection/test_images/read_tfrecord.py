import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys, getopt
sys.path.append("../../")
from object_detection.utils import label_map_util
sys.path.remove("../../")

data_path = '/media/nas/PeopleDetection/tensorflow/results/detections.tfrecord'

def add_ground_truth(image, example):
  height = int(example['image/height'].numpy())
  width = int(example['image/width'].numpy())

  xmin = example['image/object/bbox/xmin'].values.numpy()
  xmax = example['image/object/bbox/xmax'].values.numpy()
  ymin = example['image/object/bbox/ymin'].values.numpy()
  ymax = example['image/object/bbox/ymax'].values.numpy()

  xmin = xmin * width
  xmax = xmax * width
  ymin = ymin * height
  ymax = ymax * height

  bboxes = np.column_stack((xmin, ymin, xmax, ymax)).astype(dtype=int)

  label_list = example['image/object/class/text'].values.numpy()
  label = np.vectorize(lambda y: y.decode())(label_list)[0]

  for bbox in bboxes:
    y = bbox[1] - 10 if bbox[1] > 10 else bbox[1] + 10
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), [255, 255, 255], 2)
    cv2.putText(image, label, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 0], 1)

def add_detection(image, example, label_dict):
  height = int(example['image/height'].numpy())
  width = int(example['image/width'].numpy())
  
  xmin_d = example['image/detection/bbox/xmin'].values.numpy()
  xmax_d = example['image/detection/bbox/xmax'].values.numpy()
  ymin_d = example['image/detection/bbox/ymin'].values.numpy()
  ymax_d = example['image/detection/bbox/ymax'].values.numpy()

  xmin_d = xmin_d * width
  xmax_d = xmax_d * width
  ymin_d = ymin_d * height
  ymax_d = ymax_d * height

  bboxes_d = np.column_stack((xmin_d, ymin_d, xmax_d, ymax_d)).astype(dtype=int)

  labels_d = example['image/detection/label'].values.numpy()
  scores_d = example['image/detection/score'].values.numpy()

  for ix in range(len(bboxes_d)):
    bbox = bboxes_d[ix]
    label_d = labels_d[ix]
    score_d = scores_d[ix]
    y = bbox[1] - 10 if bbox[1] > 10 else bbox[1] + 10
    text = "%s: %.2f"%(label_dict[label_d], score_d)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), [0, 0, 255], 2)
    cv2.putText(image, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0, 0, 0], 1)

def main(argv):
  ground_truth = False
  try:
    opts, args = getopt.getopt(argv, "hg:")
  except getopt.GetoptError:
    print("read_tfrecord.py -g <boolean>")
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
     print('read_tfrecord.py -g <boolean>')
     sys.exit()
    elif opt == "-g":
     ground_truth = bool(arg)

  tf.enable_eager_execution()
  raw_image_dataset = tf.data.TFRecordDataset(data_path)
  feature_description = {
    'image/height': tf.FixedLenFeature((), tf.int64),
    'image/width': tf.FixedLenFeature((), tf.int64),
    'image/encoded': tf.FixedLenFeature((), tf.string),
    'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
    'image/object/class/text': tf.VarLenFeature(tf.string),
    'image/detection/bbox/xmin': tf.VarLenFeature(tf.float32),
    'image/detection/bbox/xmax': tf.VarLenFeature(tf.float32),
    'image/detection/bbox/ymin': tf.VarLenFeature(tf.float32),
    'image/detection/bbox/ymax': tf.VarLenFeature(tf.float32),
    'image/detection/label': tf.VarLenFeature(tf.int64),
    'image/detection/score': tf.VarLenFeature(tf.float32),
  }

  label_map_path = '../data/mscoco_complete_label_map.pbtxt'
  label_map_dict_inv = label_map_util.get_label_map_dict(label_map_path, use_display_name=True)
  label_map_dict = {v: k for k, v in label_map_dict_inv.items()}

  parsed_image_dataset = raw_image_dataset.map(lambda record: tf.parse_single_example(record, feature_description))

  for example in parsed_image_dataset:
    image_bytes = example['image/encoded']
    image_nparr = np.fromstring(image_bytes.numpy(), np.uint8)
    img = cv2.imdecode(image_nparr, cv2.COLOR_RGB2BGR)
    add_detection(img, example, label_map_dict)
    if ground_truth:
      add_ground_truth(img, example)
    cv2.imshow("Output", img)
    k = cv2.waitKey()
    if k == 27:
      break

if __name__ == "__main__":
   main(sys.argv[1:])