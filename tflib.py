'''process video file to detect and measure luggage carried by people in video'''
import os, sys, time, threading, cv2
import tensorflow as tf
import numpy as np
from collections import defaultdict
from multiprocessing.dummy import Queue

sys.path.append("myenv/lib/python3.5/site-packages/tensorflow/models/object_detection")
from utils import label_map_util
from utils import visualization_utils as vis_util

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

#MODEL_NAME =  'ssd_inception_v2_coco_11_06_2017'  
MODEL_NAME =  'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb' #frozen model
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')# List of strings used to label each box.
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def draw_measurements(frame,boxes):
    '''draw measurements on images -- just a test version / placeholder function for now'''
    '''
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                                thickness, display_str_list,
                                use_normalized_coordinates)
    '''
    image = Image.fromarray(np.uint8(frame)).convert('RGB')

    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    font = ImageFont.load_default()

    # Reverse list and print from bottom to top.
    print('boxes:',boxes)
    for box in boxes[0]:
        print('BOX:',box)
        xpos = box[0] * im_width
        ypos = box[1] * im_height
        
        display_str = 'measurement test'
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([
                        (xpos-2*margin, ypos - text_height - 2 * margin),
                         (xpos + text_width, ypos),
                       ], 
                        fill='red')
        draw.text((xpos - margin, ypos - text_height-2), display_str, fill='white', font=font)
        #text_bottom -= text_height - 2 * margin
        
    return image

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
    #print('BOXES:',list(zip(*boxes,*classes)))
    #print('BOXES:',list(zip(*boxes,*classes)))

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4)
    '''
    labels = {1.0:'person', 27.0:'backpack', 31.0:'handbag', 33.0:'suitcase', 3.0:'car'}
    for g in zip(*boxes,*classes):
        print('%.1f %.1f %.1f %.1f %s' %(*g[0],labels[g[-1]] if g[-1] in labels else 'unknown'))
    '''
    #image_np = draw_measurements(image_np,boxes)
    #if doing above, return image_np, not Image.fromarray as below
    return Image.fromarray(np.uint8(image_np)).convert('RGB')

def cvworker(que,commandqueue,framequeue=None):
    '''fetch frames from video file and put them into queue to send to tensorflow worker thread'''
    #cap = cv2.VideoCapture('test2.m4v')
    cap=None
    newfile=None
    newfile = commandqueue.get()

    while True:
        '''
        try:
            newfile = commandqueue.get(timeout=1/60)
            #print('newfile:',newfile)
        except:
            pass
        '''
        if newfile:
            #print('OPENING VIDEO')
            cap = cv2.VideoCapture(newfile)
            newfile = None
        if cap and cap.isOpened():
            #print('SENDING VIDEO TO TFQUEUE')
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                que.put(frame)
            else:
                #print('DONE PROCESSING VIDEO')
                cap.release()
    #return None

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

    while True:
        #time.sleep(0.1)#is this needed?
        frame = que.get()
        frame = process_frame(frame,sess,detection_graph)
        framequeue.put(frame)
        #print('added processed frame to framequeue')

    sess.close()
    #framequeue.put(None)
    #print('ending tfworker')
    #return None

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
    videoend = False
    while True:
        cvw = cv2.waitKey(1)
        if cvw & 0xFF == ord('q'): break
        if not videoend:
            print('got',frame,time.time())
            frame+=1
            print('frame:',frame)
            f=framequeue.get()
            if type(f)==type(None):
                videoend = True
                pass#whats this do
            else:
                #time.sleep(1/30) #limit to realtime
                cv2.imshow('frame',f)

    print('new took:',time.time()-start)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
