'''process video file to detect and measure luggage carried by people in video'''
import os, sys, cv2, time, threading
import tensorflow as tf
import numpy as np
from collections import defaultdict
from multiprocessing.dummy import Queue

#sys.path.append("object_detection")
from utils import label_map_util
from utils import visualization_utils as vis_util

#MODEL_NAME =  'ssd_inception_v2_coco_11_06_2017'  
MODEL_NAME =  'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb' #frozen model
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')# List of strings used to label each box.
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(2, 3) ]

def process_frame(frame,sess,detection_graph):
    '''detect objects in video frame'''
    image_np = frame
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

def cvworker(que):
    '''fetch frames from video file and put them into queue to send to tensorflow worker thread'''
    #cap = cv2.VideoCapture('test2.m4v')
    cap = cv2.VideoCapture('test2.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            que.put(frame)
        else:
            print('DONE PROCESSING VIDEO')
            que.put(None)
            break
    cap.release()
    return None

def tfworker(que,framequeue):
    ''' fetch video frames from queue and send them to object detector function,
    adding the processed result to the output frames queue, to be displayed to the user'''
    s=time.time()
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print('load time:',time.time()-s)    

    framecount = 0
    while True:
        frame = que.get()
        if type(frame)==type(None): break

        framecount+=1
        t=time.time()
        frame = process_frame(frame,sess,detection_graph)
        t2 = time.time()-t
        print('frametime:',t2)
        framequeue.put(frame)

    sess.close()
    framequeue.put(None)
    print('ending tfworker')
    return None

def main():
    '''load video, process frames, display to user'''
    tque = Queue()#(maxsize=120)
    framequeue = Queue()#(maxsize=120)
    
    cthread = threading.Thread(target=cvworker,args=(tque,))
    cthread.daemon = True
    cthread.start()
    
    tthread = threading.Thread(target=tfworker,args=(tque,framequeue))
    tthread.daemon = True #terminate testloop when user closes window
    tthread.start()  

    start=time.time()

    frame = 0
    framelist = list()
    while True:
        cvw = cv2.waitKey(1)
        if cvw & 0xFF == ord('q'): break

        print('got',frame,time.time())
        frame+=1
        f=framequeue.get()
        if type(f)==type(None):
            break
        else:
            framelist.append(f)
            #time.sleep(1/30) #limit to realtime
            cv2.imshow('frame',f)

    print('new took:',time.time()-start)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
