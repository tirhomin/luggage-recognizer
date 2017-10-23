from gevent import monkey
from gevent.wsgi import WSGIServer
monkey.patch_all()

from flask import Flask, request, render_template
from PIL import Image, ImageOps
from queue import LifoQueue, Queue
import io, time, codecs, base64, numpy
import tflib, threading, queue, time

app = Flask(__name__)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return numpy.array(image.getdata()).reshape((im_height, im_width, 3)).astype(numpy.uint8)

def imagehandler(framequeue,outqueue):
    '''update main image with analyzed image once analysis is complete'''
    while True:
        image_np, boxes, classes, scores = framequeue.get()  
        img = tflib.draw_measurements(image_np,boxes,classes,scores)
        output = Image.fromarray(numpy.uint8(img[0])).convert('RGB')
        outqueue.put(output)
        #return output
        #output.save('output.jpg')

#only store up to 3 frames so as not to get overwhelmed, dropped frames are fine
#network latency means frames will be dropped anyway, negating the usefulness of GPU
#for inference, since most of the time spent is on the network
tfque = Queue(maxsize=3) #queue for frames being sent to the neural net
framequeue = Queue(maxsize=3) #processed frames coming out of the neural net
outqueue = Queue(maxsize=3) #final labelled images to show user

cpulimit, cpus = True, 3
tthread = threading.Thread(target=tflib.tfworker,args=(tfque,framequeue,cpulimit,cpus))
tthread.daemon = True
tthread.start()

updatethread = threading.Thread(target=imagehandler,args=(framequeue,outqueue))
updatethread.daemon = True
updatethread.start()  

@app.route("/")
def home(): return render_template('demo5.html')

@app.route("/data", methods = ['GET', 'POST'])
def data():
    for f,fo in request.files.items():
        print('RF:', f, request.files[f])
        x = Image.open(io.BytesIO(base64.b64decode(fo.read())))

        arr = load_image_into_numpy_array(x)
        tfque.put(arr)
        quedata = outqueue.get()
        out = io.BytesIO()
        quedata.save(out,format='jpeg')
        
        data=out.getvalue()
        out.close()
        data = base64.b64encode(data)
        imdata="data:image/jpeg;base64,"+data.decode('ascii')
        return imdata
    
    return imdata

#app.run(debug=True, port=8000, host='0.0.0.0')
server = WSGIServer(("0.0.0.0", 8000), app)
server.serve_forever()
