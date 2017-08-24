#Installation instructions:

Create a Python 3.5 virtualenv named *tirenv*, then add tensorflow and pillow:
~~~~
pip install tensorflow pillow
~~~~

if on OSX, install [brew](http://brew.sh "brew for mac") first, then ```brew install protobuf```

then the following commands to copy some necessary files, configure path, and run protobuf compiler:

~~~~
cp -r models tirenv/lib/python3.5/site-packages/tensorflow/

export PYTHONPATH=":tirenv/lib/python3.5/site-packages/tensorflow/models:tirenv/lib/python3.5/site-packages/tensorflow/models/slim"

cd tirenv/lib/python3.5/site-packages/tensorflow/models

protoc object_detection/protos/*.proto --python_out=.
~~~~

then simply navigate back to where rgui.py is and run

~~~~
python rgui.py
~~~~

Current state of GUI:
![Alt](https://github.com/tirhomin/luggage-recognizer/blob/master/gui-wimage.jpg?raw=true "screenshot")
