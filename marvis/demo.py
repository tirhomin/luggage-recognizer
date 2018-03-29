from utils import *
from darknet import Darknet
import cv2

def demo(cfgfile, weightfile, filepath):
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    image_class_names = {20:'data/voc.names', 80:'data/coco.names'}
    class_names = load_class_names(image_class_names[m.num_classes])
 
    use_cuda = 0
    if use_cuda: m.cuda()

    #----load and analyze image----
    img = cv2.imread(filepath)
    sized = cv2.resize(img, (m.width, m.height))
    bboxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
    draw_img = plot_boxes_cv2(img, bboxes, None, class_names)
    cv2.imshow(cfgfile, draw_img)

    while True:
        #press q or spacebar to quit
        if cv2.waitKey(10) in [ord('q'), ord(' ')]:break

############################################
if __name__ == '__main__':
    demo('cfg/yolo.cfg', 'yolo.weights','demo6.jpg')
