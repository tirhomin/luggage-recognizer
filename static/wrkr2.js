/*
self.onmessage = function (event) {
  if (event.data === "Hello") {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "/testroute", false);  // synchronous request
    xhr.send(null);
    self.postMessage(xhr.responseText);
  }
};
*/
  // The width and height of the captured photo. We will set the
  // width to the value defined here, but the height will be
  // calculated based on the aspect ratio of the input stream.
  var PAUSE = false;
  var width = 400;    // We will scale the photo width to this
  var height = 0;     // This will be computed based on the input stream

  // |streaming| indicates whether or not we're currently streaming
  // video from the camera. Obviously, we start at false.

  var streaming = false;

  // The various HTML elements we need to configure or control. These
  // will be set by the startup() function.

  var video = null;
  var canvas = null;
  var photo = null;
  var startbutton = null;

  function pausehandler(){
    if(PAUSE){
      video.play();
      PAUSE = false; 
      pbtn = document.getElementById('PAUSE').firstChild;
      pbtn.nodeValue = "PAUSE";
      console.log('unpaused');
    }
    else{
      video.pause();
      //takepicture();
      PAUSE = true; 
      pbtn = document.getElementById('PAUSE').firstChild;
      pbtn.nodeValue = "UNPAUSE";
      console.log('paused');
    }
    return 1;}

  function startup() {
      console.log('inside startup function');
    pbtn = document.getElementById('PAUSE');
    pbtn.addEventListener('click', pausehandler, false);    
    
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    photo = document.getElementById('photo');
    startbutton = document.getElementById('startbutton');

    navigator.getMedia = ( navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia);
    navigator.getMedia({video: true,audio: false},
      function(stream) {
        if (navigator.mozGetUserMedia) {video.mozSrcObject = stream;}
        else {
          var vendorURL = window.URL || window.webkitURL;
          video.src = vendorURL.createObjectURL(stream);
        }
        video.play();
      },
      function(err) {console.log("An error occured! " + err);}
    );

    video.addEventListener('canplay', function(ev){
      if (!streaming) {
        height = video.videoHeight / (video.videoWidth/width);     
        // Firefox currently has a bug where the height can't be read from
        // the video, so we will make assumptions if this happens.
        if (isNaN(height)) {height = width / (4/3);}
        video.setAttribute('width', width);
        video.setAttribute('height', height);
        canvas.setAttribute('width', width);
        canvas.setAttribute('height', height);
        streaming = true;
      }
    }, false);

    window.setInterval(function(){takepicture();},1000)
    clearphoto();
  }

  // Fill the photo with an indication that none has been captured.
  // perhaps unnecessary now and can probably be deleted
  function clearphoto() {
    var context = canvas.getContext('2d');
    context.fillStyle = "#AAA";
    context.fillRect(0, 0, canvas.width, canvas.height);
  }
  
  // SEND IMAGE TO SERVER, GET RESPONSE IMAGE, DISPLAY IMAGE
  function sendPic(blob) {
            var data = new FormData();
            data.append("imagefile", blob, ("camera.png"));
            var oReq = new XMLHttpRequest();
            oReq.open("POST", "/data");
            oReq.send(data);
                oReq.onload = function(oEvent) {
                    if(PAUSE){return 0;}
                    if (oReq.status == 200) {
                        console.log("Uploaded");
                        console.log(oReq.response);
                        photo.setAttribute('src', oReq.response);
                    } else {
                        console.log("Error " + oReq.status + " occurred uploading your file.");
                    }
                };
            }

  // TAKE A PHOTO FROM WEBCAM; 
  // SEND WEBCAM IMAGE TO SERVER;
  // WAIT FOR RESPONSE WITH A PROCESSED IMAGE;
  // UPDATE CANVAS WITH THAT IMAGE
  // IF NO PHOTO WAS CAPTURED, CLEAR THE FRAME TO SHOW USER IT IS FROZEN / WEBCAM FAILURE
  function takepicture() {
    if(PAUSE){console.log('doing nothing while paused'); return 0;}
    else{
        //console.log('not paused, proceeding');
    }
    
    
    var context = canvas.getContext('2d');
    if (width && height) {
      canvas.width = width;
      canvas.height = height;
      context.drawImage(video, 0, 0, width, height);
    
      var iformat = 'image/jpeg';
      var data = canvas.toDataURL(iformat);

      var base64PrefixLength = ('data:'+iformat+';base64,').length;
      var bdata = data.slice(base64PrefixLength);
      var blob = new Blob([bdata,], {type: iformat});
      sendPic(blob);

    } else {
      clearphoto();
    }
  }

//-------------------------------------
  self.onmessage = function (event) {
    if (event.data === "start the worker") {
        console.log('starting worker')
        startup();
    }
  };
  