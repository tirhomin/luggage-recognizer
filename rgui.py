import tflib
import tkinter as tk
import os, threading, queue, time, copy #stdlib
from multiprocessing.dummy import Queue
from PIL import ImageTk, Image, ImageDraw, ImageFont
from tkinter import filedialog, font
import numpy as np

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def updatethread(self):
    '''update main image with analyzed image once analysis is complete'''
    while True:       
        #print("updatethread")
        image_np, boxes, classes, scores = self.framequeue.get()  
        img = tflib.draw_measurements(image_np,boxes,classes,scores)
        image_np = img[0] 
        bboxes = img[1]
        self.tfimg = Image.fromarray(np.uint8(image_np)).convert('RGB')
        self.tfimg.thumbnail(self.VIDSIZE,Image.ANTIALIAS)
        self.tfimg = ImageTk.PhotoImage(self.tfimg)
        self.mainimg = self.tfimg
        self.origimg = image_np
        print('\n\n------------\nIDS: ',id(self.origimg), id(self.mainimg), id(self.tfimg))
        self.bboxes = bboxes
        self.videolabel.configure(image=self.mainimg, width=self.VIDSIZE[0], height=self.VIDSIZE[1])
        #TODO TRY MOVING JUST THE LAST STATEMENT INTO MAIN THREAD

class BOTTOMBAR(object):
    ''' bottom toolbar beneath main image display'''
    def __init__(oself,self):
        self.fileframe = tk.Frame(self.bottomframe,width=self.TILESIZE*5, height=self.TILESIZE*4)#, bg="#0F0");
        self.fileframe.pack(side=tk.TOP, anchor="w") 
        self.filebtn=tk.Button(self.fileframe,text='ADD IMAGE OR VIDEO', anchor="w", command=self.change_file)
        self.filebtn['font'] = self.bfont
        self.filebtn.pack(side=tk.LEFT)

        self.filename = tk.StringVar()
        self.filelabel = tk.Label(self.fileframe, textvariable=self.filename, relief='sunken')#image=img)
        self.filelabel['font'] = self.pfont
        self.filelabel.pack(side=tk.LEFT)
        self.filename.set('NO FILE SELECTED')

        self.dividerframe = tk.Frame(self.bottomframe,width=64,height=6)
        self.dividerframe.pack(side=tk.TOP)

        self.bottombtnframe = tk.Frame(self.bottomframe)
        self.bottombtnframe.pack(side=tk.TOP, anchor="w")

        self.originalbtn=tk.Button(self.bottombtnframe,text='show original file', command=self.show_original)
        self.originalbtn['font'] = self.bfont
        self.originalbtn.pack(side=tk.LEFT)

        self.analyzedbtn=tk.Button(self.bottombtnframe,text='show analyzed file', command=self.show_analyzed)
        self.analyzedbtn['font'] = self.bfont
        self.analyzedbtn.pack(side=tk.LEFT)

        self.fillerframe = tk.Frame(self.bottomframe,width=self.TILESIZE,height=self.TILESIZE)#, bg="#C00")
        self.fillerframe.pack(side=tk.TOP)


class RIGHTBAR(object):
    ''' right tool bar / previous results '''
    def __init__(oself,self):
        self.histimg1 = Image.new('RGB',self.THUMBSIZE)
        self.histimg1 = ImageTk.PhotoImage(self.histimg1)
        self.histpath1=None

        self.histimg2 = Image.new('RGB', self.THUMBSIZE)
        self.histimg2 = ImageTk.PhotoImage(self.histimg2)
        self.histpath2=None

        self.histimg3 = Image.new('RGB', self.THUMBSIZE)
        self.histimg3 = ImageTk.PhotoImage(self.histimg3)
        self.histpath3=None
        self.histlabel1 = tk.Label(self.rightframe, image=self.histimg1)
        self.histlabel1.bind("<Button-1>",lambda e:self.load_previous(1))

        self.histlabel2 = tk.Label(self.rightframe, image=self.histimg2)
        self.histlabel2.bind("<Button-1>",lambda e:self.load_previous(2))

        self.histlabel3 = tk.Label(self.rightframe, image=self.histimg3)
        self.histlabel3.bind("<Button-1>",lambda e:self.load_previous(3))

        self.histlabel1.pack(side=tk.TOP,anchor="c")
        self.histlabel2.pack(side=tk.TOP)#, anchor="n",fill=tk.X, expand=tk.YES)
        self.histlabel3.pack(side=tk.TOP)

        #img0 is a placeholder to fill UI space
        self.histimg0 = Image.new('RGBA', self.THUMBSIZE)
        self.histimg0 = ImageTk.PhotoImage(self.histimg0)
        self.histlabel0 = tk.Label(self.rightframe, image=self.histimg0)
        self.histlabel0.pack(side=tk.TOP, anchor="n",fill=tk.BOTH, expand=1)

class VIDEOBOX(object):
    '''main image view or video feed'''
    def __init__(oself, self):
        #THE IMAGE TO PROCESS
        self.rawimg = Image.new('RGB', self.VIDSIZE)
        self.rawimg = ImageTk.PhotoImage(self.rawimg)

        #THE PROCESSED IMAGE
        self.tfimg = Image.new('RGB', self.VIDSIZE)
        self.tfimg = ImageTk.PhotoImage(self.tfimg)

        #THE DISPLAYED IMAGE COULD BE EITHER OF THE ABOVE
        self.mainimg = Image.new('RGB', self.VIDSIZE)
        self.mainimg = ImageTk.PhotoImage(self.mainimg)

        self.origimg = None
        self.bboxes = None

        self.videolabel = tk.Label(self.videoframe, image=self.mainimg)#text="IMAGE OR VIDEO NOT YET LOADED")#image=img)
        self.videolabel.pack(side=tk.TOP)
        self.videolabel.configure(image=self.mainimg, width=self.VIDSIZE[0], height=self.VIDSIZE[1])
        self.videolabel.bind("<Button-1>", self.mouse_function)

class GUI(object):
    '''the main GUI window'''
    def __init__(self, root):
        self.TILESIZE = 128
        thumbscale = 1.95
        self.THUMBSIZE =  (int(self.TILESIZE*thumbscale),int(self.TILESIZE*thumbscale/1.7777777))
        self.VIDSIZE = (self.TILESIZE*5,int(self.TILESIZE*5/1.777777))
        self.root = root
        self.root.title("LUGGAGE RECOGNIZER")
        self.root.geometry("%dx%d" %(self.TILESIZE*7,int(self.TILESIZE*3.5)))
        self.root.protocol("WM_DELETE_WINDOW",self.shutdown)

        self.bfont = font.Font(family='Helvetica', size=12, weight='bold')
        self.pfont = font.Font(family='Helvetica', size=12)
        self.mfont = ImageFont.truetype(font='roboto.ttf', size=18)

        self.leftframe = tk.Frame(self.root,width=self.TILESIZE*5, height=self.TILESIZE*4)#, bg="#0F0"); 
        self.leftframe.pack(side=tk.LEFT)
       
        self.rightframe = tk.Frame(self.root,width=self.TILESIZE*2, height=self.TILESIZE*4)#, bg="#FFF")
        self.rightframe.pack(side=tk.LEFT)
       
        self.videoframe = tk.Frame(self.leftframe,width=self.VIDSIZE[0], height=self.VIDSIZE[1])#, bg="#222")
        self.videoframe.pack(side=tk.TOP,fill=None, expand=False)

        self.bottomframe = tk.Frame(self.leftframe,width=self.VIDSIZE[0],height=self.TILESIZE)#, bg="#E80")
        self.bottomframe.pack(side=tk.LEFT)

        self.videobox = VIDEOBOX(self)
        self.bottombar = BOTTOMBAR(self)
        self.rightbar = RIGHTBAR(self)

    def shutdown(self):
        '''cleanly exit threads, databases, serial ports, etc? TODO/if necessary'''
        #time.sleep(0.1)
        self.root.destroy()

    def show_original(self):
        '''show original image w/o analysis labels'''
        self.videolabel.configure(image=self.rawimg, width=self.VIDSIZE[0], height=self.VIDSIZE[1])

    def show_analyzed(self):
        '''show image w/ analysis labels'''
        self.videolabel.configure(image=self.tfimg, width=self.VIDSIZE[0], height=self.VIDSIZE[1])

    def load_previous(self,num):
        '''load a historical image as chosen from right toolbar'''
        print('LOADPREV:',num)
        path = getattr(self,'histpath'+str(num))
        print('loading histpath:',path)
        self.set_image(path)

    def mouse_function(self,e):
        print('EVENT:',e,e.x,e.y)
        print('\n\n------------\nZZZIDS: ',id(self.origimg), id(self.mainimg), id(self.tfimg))

        if self.origimg:
            image = self.origimg
            #MOUSE POSITION RELATIVE TO IMAGE
            print('vs',self.VIDSIZE)
            dispwidth = self.mainimg.width() #width of scaled image on screen
            dispheight = self.mainimg.height()
            scale = image.size[0] / dispwidth
            print('iw, dw, s', image.size[0],dispwidth,scale)
            offsetx = (self.VIDSIZE[0] - dispwidth)/2
            offsety = (self.VIDSIZE[1] - dispheight)/2
            rx = e.x - offsetx
            ry = e.y - offsety
            print('rx,ry',rx,ry,offsetx,offsety)
            rx *= scale
            ry *= scale
            print('scaled rx,ry',rx,ry)
            
            #---------------------------
            newimg = copy.copy(image)
            draw = ImageDraw.Draw(newimg)
            for box in self.bboxes:
                print('BOX:',box)
                xpos,ypos,x2pos,y2pos = box
                if (xpos < rx < x2pos) and (ypos < ry < y2pos):
                    print("IN BBOX!")
                    display_str = 'SIZE: %.0fx%.0f CM' %(x2pos-xpos, y2pos-ypos)
                    text_width, text_height = self.mfont.getsize(display_str)
                    #margin = np.ceil(0.05 * text_height)
                    '''
                    draw.rectangle([
                                    (xpos, ypos),
                                    (x2pos, y2pos),
                                ], 
                                    fill='#F005')
                    '''
                    draw.rectangle([
                            (xpos, ypos),
                            (xpos + text_width, ypos + text_height+4),
                        ], 
                            fill='red')
                    draw.line([(xpos,ypos), (xpos, y2pos), (x2pos, y2pos),
                        (x2pos, ypos), (xpos, ypos)], width=7, fill='#F00')
                    display_str = 'SIZE: %.0fx%.0f CM' %(x2pos-xpos, y2pos-ypos)
                    text_width, text_height = self.mfont.getsize(display_str)
                    margin = np.ceil(0.05 * text_height)
                    draw.text((xpos, ypos), display_str, fill='white', font=self.mfont)

            newimg.thumbnail(self.VIDSIZE,Image.ANTIALIAS)
            self.mainimg=ImageTk.PhotoImage(newimg)
            
            self.videolabel.configure(image=self.mainimg, width=self.VIDSIZE[0], height=self.VIDSIZE[1])
            
    def set_image(self,fname):
        '''set an image to the main window from a filepath to image'''
        openedimg = Image.open(fname)
        self.tque.put(load_image_into_numpy_array(openedimg))
        self.mainimg = openedimg
        self.mainimg.thumbnail(self.VIDSIZE,Image.ANTIALIAS)
        self.mainimg = ImageTk.PhotoImage(self.mainimg)
        self.rawimg = self.mainimg
        self.videolabel.configure(image=self.mainimg, width=self.VIDSIZE[0], height=self.VIDSIZE[1])

    def clear_queues(self):
        for q in [self.tque,self.framequeue]:
            q.mutex.acquire()
            q.queue.clear()
            q.all_tasks_done.notify_all()
            q.unfinished_tasks = 0
            q.mutex.release()

    def change_file(self):
        '''prompt user to input new image or video file'''
        images = ['jpg','jpeg','png']
        videos = ['mp4','m4v']
        allowed_names = videos + images
        isimage = False

        fname = filedialog.askopenfilename()
        if not fname:
            print('no file selected')
            return None
        else:
            ext = fname.split('.')[-1]
        
        if not ext.lower() in allowed_names:
            print('unknown filetype, must be jpg, jpeg, png, mp4, or m4v')
            return None
        if ext.lower() in images:
            isimage = True

        self.clear_queues()
        #stop adding stuff to queue
        if self.videosupport: self.cvcommandqueue.put(0xDEAD)

        self.filename.set(os.path.basename(fname))
        #updatepaths to historical images
        self.histpath3 = self.histpath2
        self.histpath2 = self.histpath1
        self.histpath1 = fname

        if isimage:
            self.set_image(fname)
            #update historical images on right toolbar
            self.histimg3=self.histimg2
            self.histimg2=self.histimg1
            tempimg = Image.open(fname)
            tempimg.thumbnail(self.THUMBSIZE,Image.ANTIALIAS)
            self.histimg1 = ImageTk.PhotoImage(tempimg)
            #update recent images:
            self.histlabel1.configure(image=self.histimg1)
            self.histlabel2.configure(image=self.histimg2)
            self.histlabel3.configure(image=self.histimg3)

        else:
            #VIDEO
            print('video integration in dev')
            if self.videosupport:
                print('video support enabled')
                self.cvcommandqueue.put(fname)
                print('added video file to cv command queue')
                pass
        

    def run(self,videosupport=False,debug=False,cpulimit=False):
        '''start the application!'''
        self.tque = Queue()#(maxsize=120)
        self.framequeue = Queue()#(maxsize=120)
        self.cvcommandqueue = Queue()
        self.videosupport = videosupport

        self.cvthread = None
        if self.videosupport:
            self.cvthread = threading.Thread(target=tflib.cvworker,args=(self.tque,self.cvcommandqueue,cpulimit))
            self.cvthread.daemon = True
            self.cvthread.start()  
        self.tthread = threading.Thread(target=tflib.tfworker,args=(self.tque,self.framequeue,cpulimit))
        self.tthread.daemon = True
        self.tthread.start()  

        #'''
        self.updatethread = threading.Thread(target=updatethread,args=(self,))
        self.updatethread.daemon = True
        self.updatethread.start()  
        #'''
        self.debug=debug
        if self.debug:
            self.root.after(500,self.set_image,'demo5.jpg')
        self.root.mainloop()
#----------------------

if __name__ == '__main__':
    gui = GUI(tk.Tk())
    gui.run(videosupport=True,debug=True,cpulimit=True)