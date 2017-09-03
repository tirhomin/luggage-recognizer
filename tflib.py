'''process video file to detect and measure luggage carried by people in video'''
import os, sys, time, threading, platform, cv2
import tensorflow as tf
import numpy as np
from collections import defaultdict
from multiprocessing.dummy import Queue

operatingsystem = platform.system()
if operatingsystem=='Darwin':
    #for our OSX config
    sys.path.append("newenv/lib/python3.6/site-packages/tensorflow/models/object_detection")
elif operatingsystem=='Linux':
    #for our Linux config
    sys.path.append("newenv/lib/python3.5/site-packages/tensorflow/models/object_detection")
else:
    print('ERROR: virtualenv not found?')

from utils import label_map_util
from utils import visualization_utils as vis_util
from PIL import ImageTk, Image, ImageDraw, ImageFont

#MODEL_NAME =  'ssd_inception_v2_coco_11_06_2017'  
MODEL_NAME =  'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb' #frozen model
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')# List of strings used to label each box.
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
mfont = ImageFont.truetype(font='roboto.ttf', size=18)

def draw_measurements(frame, boxes, classes, scores):
    '''draw measurements on images -- just a test version / placeholder function for now'''
    '''
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                                thickness, display_str_list,
                                use_normalized_coordinates)
    '''
    image = Image.fromarray(np.uint8(frame)).convert('RGB')

    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    #font = ImageFont.load_default()

    # Reverse list and print from bottom to top.
    #print('boxes:',boxes)
    labels = {1.0:'person', 27.0:'backpack', 31.0:'handbag', 33.0:'suitcase', 3.0:'car'}
    bboxes = list()
    for group in zip(*boxes,*classes,*scores):#boxes[0]:
        box = group[0]
        score = group[-1]
        if score >= 0.5 and group[-2] in (27.0,31.0,33.0):#backpack,handbag,suitcase
            #print('group:',group)
            xpos = box[1] * im_width
            ypos = box[0] * im_height
            
            x2pos = box[3] * im_width
            y2pos = box[2] * im_height

            bboxes.append((xpos,ypos,x2pos,y2pos))

            '''
            display_str = 'SIZE: %.0fx%.0f CM' %(x2pos-xpos, y2pos-ypos)
            text_width, text_height = mfont.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle([
                            (xpos, ypos),
                            (xpos + text_width, ypos + text_height),
                        ], 
                            fill='red')

            draw.text((xpos, ypos), display_str, fill='white', font=mfont)
            '''
            #text_bottom -= text_height - 2 * margin
    
    return (image,bboxes)

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
    #print('classes',dir(classes))
    t1=time.time()
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    print('framtime:',time.time()-t1)
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
    #print('boxes:',boxes)

    for g in zip(*boxes,*classes,*scores):
        if g[-1]>=0.5:
            print('%.1f %.1f %.1f %.1f %s %s' %(*g[0],labels[g[-2]] if g[-2] in labels else 'unknown', str(g[-1])))
    #'''
    return (image_np, boxes, classes, scores)
    #image_np = draw_measurements(image_np,boxes,classes,scores)
    #if doing above, return image_np, not Image.fromarray as below
    #return Image.fromarray(np.uint8(image_np)).convert('RGB')

def cvworker(que,commandqueue,framequeue=None,cpulimit=False):
    '''fetch frames from video file and put them into queue to send to tensorflow worker thread'''
    newfile = commandqueue.get()
    if not newfile==0xDEAD:
        cap = cv2.VideoCapture(newfile)
    else:
        cap = None

    while True:
        if cpulimit:time.sleep(1/30)
        try:
            newfile = commandqueue.get(timeout=1/50)
            if newfile == 0xDEAD:
                cap.release()
            else:
                cap = cv2.VideoCapture(newfile)     
        except:
            pass

        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                que.put(frame)
            else:
                cap.release()

def tfworker(que,framequeue,cpulimit=False):
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
            if cpulimit:
                config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
                                        allow_soft_placement=True, 
                                        device_count = {'CPU': 1})
                sess = tf.Session(graph=detection_graph,config=config)
            else:
                sess = tf.Session(graph=detection_graph)

    while True:
        if cpulimit: time.sleep(1/30)
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
