from gevent import monkey
from gevent.wsgi import WSGIServer
monkey.patch_all()

from flask import Flask, request, render_template
from PIL import Image, ImageOps
from queue import LifoQueue, Queue
import io, time, codecs, base64, numpy
import threading, queue, time
from marvis import torchlib as nnlib
app = Flask(__name__)
import os
print('OS1:',os.getcwd())
def load_image_into_numpy_array(image):
    '''convert PIL image data into numpy array for manipulation by TensorFlow'''
    (im_width, im_height) = image.size
    return numpy.array(image.getdata()).reshape((im_height, im_width, 3)).astype(numpy.uint8)

def imagehandler(framequeue,outqueue):
    '''take numpy array of webcam image and process with tensorflow to
    detect objects and take measurements, add results to output queue'''
    while True:
        #these three lines for tensorflow version
        #image_np, boxes, classes, scores = framequeue.get()  
        #img = nnlib.draw_measurements(image_np,boxes,classes,scores)
        #output = Image.fromarray(numpy.uint8(img[0])).convert('RGB')
        
        #these two lines for pytorch version
        image_np = framequeue.get()
        output = Image.fromarray(numpy.uint8(image_np)).convert('RGB')
       
        outqueue.put(output)

#only store up to 3 frames so as not to get overwhelmed, dropped frames are fine
#network latency means frames will be dropped anyway, negating the usefulness of GPU
#for inference, since most of the time spent is on the network
tfque = Queue(maxsize=3) #queue for frames being sent to the neural net
framequeue = Queue(maxsize=3) #processed frames coming out of the neural net
outqueue = Queue(maxsize=3) #final labelled images to show user

#use only 3 of our 4 CPUs for tensorflow 
#so the web server remains responsive on the 4th CPU
cpulimit, cpus = True, 3

#tthread is the tensorflow or torch thread

#FASTEST & LOWEST ACCURACY
#wkrconfig={'cfgfile':'marvis/cfg/tiny-yolo.cfg','weightfile':'marvis/bin/tiny-yolo.weights','cuda':0}

#LESS FAST, MUCH MORE ACCURATE (plenty fast on GPU though)
wkrconfig={'cfgfile':'marvis/cfg/yolo.cfg','weightfile':'marvis/bin/yolo.weights','cuda':0}
tthread = threading.Thread(target=nnlib.tfworker,args=(tfque,framequeue,cpulimit,cpus,wkrconfig))
tthread.daemon = True
tthread.start()

#updatethread is fetching images from input queue, feeding them to tensorflow thread
#and then adding the results to an output queue for return to the client
updatethread = threading.Thread(target=imagehandler,args=(framequeue,outqueue))
updatethread.daemon = True
updatethread.start()  

@app.route("/")
def home():
    '''main page / Web UI for webcam'''
    return render_template('1.5.html')

@app.route("/data", methods = ['GET', 'POST'])
def data():
    '''accept AJAX request containing webcam image, respond with processed image'''
    #process that image
    for f,fo in request.files.items():
        #print('RF:', f, request.files[f])
        #decode base64 jpeg from client request, convert to PIL Image
        x = Image.open(io.BytesIO(base64.b64decode(fo.read())))

        #convert image to numpy array and insert into tensorflow queue
        arr = load_image_into_numpy_array(x)
        tfque.put(arr)

        #get the next available processed frame from the output queue
        #note this may not be the image we inserted into the tensorflow queue
        #that image may still be awaiting processing, this image is instead simply
        #the next available one, which may have been processed some milliseconds ago
        quedata = outqueue.get()
        out = io.BytesIO()
        quedata.save(out,format='jpeg')
        
        #convert to base64 to respond to client 
        data=out.getvalue()
        out.close()
        data = base64.b64encode(data)

        #add data type so the browser can simply render the base64 data as an image on a canvas
        imdata="data:image/jpeg;base64,"+data.decode('ascii')
        return imdata
    
    return 'no webcam image provided'

@app.route('/testroute')
def testroute():
    return 'test route returned OK'
#debug server which auto-reloads for live changes during development
#app.run(debug=True, port=8080, host='0.0.0.0')

#run the server with gevent to support multiple clients on AWS
server = WSGIServer(("0.0.0.0", 8080), app)
server.serve_forever()
