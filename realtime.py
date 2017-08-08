import os, sys, cv2
import tensorflow as tf
import numpy as np
from collections import defaultdict
from io import StringIO
from PIL import Image

#sys.path.append("object_detection")
from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb' #frozen model
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')# List of strings used to label each box.
NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(2, 3) ]

def process_frame(frame,sess):
    image_np = )
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4)
    return image_np

cap = cv2.VideoCapture('test2.mp4')#vtest.avi')
#out = cv2.VideoWriter('output.avi', -1, 20.0, (640,360))
ret, frame = cap.read()
nextframe = frame
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while(cap.isOpened()):
            cvw = cv2.waitKey(1)
            if  True:#cvw & 0xFF == ord('a'):
                ret, frame = cap.read()
                nextframe = process_frame(frame,sess)
                #cv2.imwrite('frame1.jpg',nextframe)
            if cvw & 0xFF == ord('q'):
                break
            '''if ret==True:
                frame = cv2.flip(frame,0)
                cv2.imwrite('testframe.jpg',frame)'''
            cv2.imshow('frame',nextframe)

cap.release()
cv2.destroyAllWindows()