######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Soumyadip Majumder
# Date: 27/07/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py
## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import glob

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
#IMAGE_NAME = 'coconuts1.jpg'

#Import multiple images
IMAGE_PATHS = [cv2.imread(file) for file in glob.glob(os.path.join(CWD_PATH,"fish_pics\\*.jpg"))]
#print(glob.glob(os.path.join(CWD_PATH,"fish_pics\\*.jpg")))

#for file in glob.glob(os.path.join(CWD_PATH,"faces\\*.jpg")):
#     print(file,"  ",os.path.getctime(file)) 

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
#PATH_TO_IMAGE = os.path.join(CWD_PATH,"Coconuts\\",IMAGE_PATHS[0])
#print(CWD_PATH)
#print(PATH_TO_IMAGE)

# Number of classes the object detector can identify
NUM_CLASSES = 4

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `1`, we know that this corresponds to `coconut`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
#image = IMAGE_PATHS[1]
#image_expanded = np.expand_dims(image, axis=0)

count = []
for i in range(0,len(IMAGE_PATHS)):
   count.append(0)

print(count)
file = glob.glob(os.path.join(CWD_PATH,"fish_pics\\*.jpg"))
for i in range(0,len(IMAGE_PATHS)):

    FILENAME_TIMESTAMP= "\n" + str(file[i]) +"#" + str(time.asctime(time.localtime(time.time()))) + "#"
    with open("C:\\Fish_Detection\\models\\research\\object_detection\\Fish_Detection_Output.txt", "a") as oup1:
                oup1.write(FILENAME_TIMESTAMP)
    print(FILENAME_TIMESTAMP)

    image_expanded = np.expand_dims(IMAGE_PATHS[i], axis=0)
    #print(image_expanded.shape)
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
        IMAGE_PATHS[i],
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.65)


    # All the results have been drawn on image. Now display the image.
    #Print the total count of Coconut
    final_score = np.squeeze(scores)    
    #count[i] = 0
    for j in range(100):
        if scores is None or final_score[j] > 0.5:
            count[i] = count[i] + 1
    
    print(count)
    fish_det="Fish Count: "+str(count[i])
    #cv2.putText(IMAGE_PATHS[i],fish_det, (170,310), cv2.FONT_HERSHEY_SIMPLEX, .5,(255,0,255),2,cv2.LINE_AA)
    cv2.imwrite( "Fish_Detected_Imgs/pic"+ str(i) + ".jpg", IMAGE_PATHS[i] )
    cv2.imshow('Object detector', IMAGE_PATHS[i])
    


# Press any key to close the image
    cv2.waitKey(0)

# Clean up
#cv2.destroyAllWindows()

print("Total Fish Count:", sum(count[]))
text_file = open("C:\\Fish_Detection\\models\\research\\object_detection\\Fish_Count_Output.txt", "w")
text_file.write("Coconut Count: %s" % sum(count[]))
text_file.close()
