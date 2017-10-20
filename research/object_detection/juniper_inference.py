import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import glob
import math
import time
from lxml import etree as ET

from PIL import Image
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt

from utils import label_map_util
from utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

PATH_TO_CKPT = "C:\\Users\\schriver\\Code\\juniper\\archive\\rcnn_resnet101_09112017\\graph\\frozen_inference_graph.pb"

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

PATH_TO_LABELS = "C:\\Users\\schriver\\Code\\juniper\\data\\label map juniper ponderosa large_shrub.pbtxt"
NUM_CLASSES = 3

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

print(categories)
print(category_index)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = "C:\\Users\\schriver\\Code\\juniper\\data\\test\\image input"
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, "*.jpg"))

PATH_TO_TEST_IMAGES_OUTPUT = "C:\\Users\\schriver\\Code\\juniper\\data\\test\\image output"
PATH_TO_XML_OUTPUT = "C:\\Users\\schriver\\Code\\juniper\\data\\test\\xml output"

# Size, in inches, of the output images.
IMAGE_SIZE = (4, 4)


with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		print(time.time())
		for image_path in TEST_IMAGE_PATHS:
			_, image_dirname = os.path.split(image_path)
			image_filename,_ = os.path.splitext(image_dirname) 

			annotation = ET.Element("annotation")
			
			ET.SubElement(annotation, "folder").text = "jpg"
			ET.SubElement(annotation, "filename").text = image_filename

			size = ET.SubElement(annotation, "size")
			ET.SubElement(size, "width").text = "100"
			ET.SubElement(size, "height").text = "100"
			ET.SubElement(size, "depth").text = "3"

			ET.SubElement(annotation, "segmented").text = "0"

			image = Image.open(image_path)
			# the array based representation of the image will be used later in order to prepare the
			# result image with boxes and labels on it.
			image_np = load_image_into_numpy_array(image)
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			# Each box represents a part of the image where a particular object was detected.
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			# Actual detection.
			(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
			for box,score,classification in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):
				if score > 0.5:
					obj = ET.SubElement(annotation, "object")
					ET.SubElement(obj, "name").text = category_index[classification]['name']
					ET.SubElement(obj, "pose").text = "Unspecified"
					ET.SubElement(obj, "truncated").text = "0"
					ET.SubElement(obj, "difficult").text = "0"

					bndbox = ET.SubElement(obj, "bndbox")
					ET.SubElement(bndbox, "ymin").text = str(int(math.floor(box[0] * 100)))
					ET.SubElement(bndbox, "xmin").text = str(int(math.floor(box[1] * 100)))
					ET.SubElement(bndbox, "ymax").text = str(int(math.floor(box[2] * 100)))
					ET.SubElement(bndbox, "xmax").text = str(int(math.floor(box[3] * 100)))
			
			tree = ET.ElementTree(annotation)
			tree.write(os.path.join(PATH_TO_XML_OUTPUT, image_filename + '.xml'), pretty_print=True)
	
			# Visualization of the results of a detection.
			vis_util.visualize_boxes_and_labels_on_image_array(
				image_np,
				np.squeeze(boxes),
				np.squeeze(classes).astype(np.int32),
				np.squeeze(scores),
				category_index,
				use_normalized_coordinates=True,
				max_boxes_to_draw=100,
				min_score_thresh=0.5,
				line_thickness=1)

			export_path = os.path.join(PATH_TO_TEST_IMAGES_OUTPUT, image_filename + '.jpg')
			vis_util.save_image_array_as_png(image_np, export_path)
			
		print(time.time())
