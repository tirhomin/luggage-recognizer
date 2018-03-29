'''process video file to detect and measure luggage carried by people in video'''
import os, sys, time, threading, platform, numpy, cv2
from collections import defaultdict
from multiprocessing.dummy import Queue
from .darknet import Darknet
from .utils import *
    
def cvworker(que,commandqueue,framequeue=None,cpulimit=False):
    '''fetch frames from video file and put them into queue to send to torch worker thread'''
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

def tfworker(que,framequeue,cpulimit=False,cpus=1,config={'cfgfile':'marvis/yolo.cfg','weightfile':'marvis/yolo.weights','cuda':0}):
    ''' fetch video frames from queue and send them to object detector function,
    adding the processed result to the output frames queue, to be displayed to the user'''
    yolomodel = Darknet(config['cfgfile'])
    yolomodel.print_network()
    yolomodel.load_weights(config['weightfile'])

    image_class_names = {20:'marvis/data/voc.names', 80:'marvis/data/coco.names'}
    class_names = load_class_names(image_class_names[yolomodel.num_classes])
 
    if config['cuda']: yolomodel.cuda()
    #img = cv2.imread(filepath)

    while True:
        if cpulimit: time.sleep(1/30)
        frame = que.get()
        #------------
        if True:
            sized = cv2.resize(frame, (yolomodel.width, yolomodel.height))
            bboxes = do_detect(yolomodel, sized, 0.5, 0.4, use_cuda = config['cuda'])
            draw_img = plot_boxes_cv2(frame, bboxes, None, class_names)
            framequeue.put(draw_img)
        #-------------
        #frame = process_frame(frame,sess,detection_graph)
        #framequeue.put(frame)



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
