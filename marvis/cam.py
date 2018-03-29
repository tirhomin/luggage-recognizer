from utils import *
from darknet import Darknet
import cv2

def demo(cfgfile, weightfile):
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    image_class_names = {20:'data/voc.names', 80:'data/coco.names'}
    class_names = load_class_names(image_class_names[m.num_classes])
 
    use_cuda = 0
    if use_cuda: m.cuda()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1)

    while True:
        res, img = cap.read()
        if res:
            sized = cv2.resize(img, (m.width, m.height))
            bboxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
            print('------')
            draw_img = plot_boxes_cv2(img, bboxes, None, class_names)
            cv2.imshow(cfgfile, draw_img)

            #press q or spacebar to quit
            if cv2.waitKey(10) in [ord('q'), ord(' ')]:break
        else:
             print("Unable to read image")
             exit(-1) 

if __name__ == '__main__':
    demo('cfg/yolo.cfg', 'yolo.weights')
