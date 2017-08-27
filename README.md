#Installation instructions:

if on OSX, install [brew](http://brew.sh "brew for mac") first, then ```brew install protobuf```

then the following commands to copy some necessary files and set up some configuration stuff:

assuming a brand new copy of the repo in a directory named ```/luggage-recognizer``` somewhere,

open a terminal in that directory, and ```virtualenv -p python3.5 myenv```

then, activate it by ```source myenv/bin/activate```

then, ```pip install tensorflow pillow```

then, ```cp -r models myenv/lib/python3.5/site-packages/tensorflow/``` (or do that manually if it doesn't work)

then do this, including the quotation marks: ```export PYTHONPATH=":myenv/lib/python3.5/site-packages/tensorflow/models:myenv/lib/python3.5/site-packages/tensorflow/models/slim"```

then descend into the models directory: ```cd myenv/lib/python3.5/site-packages/tensorflow/models```

there, you'll run the protobuf compiler: ```protoc object_detection/protos/*.proto --python_out=.```

then you can go back up to the luggage-recognizer directory: ```cd ../../../../../../```

then it should work by typing ```python rgui.py```

Current state of GUI:![Alt](gui-wimage.jpg?raw=true "screenshot")