#Installation instructions:

Create a Python 3.5 virtualenv named *tirenv*, then do the following:
~~~~
cp -r models tirenv/lib/python3.5/site-packages/tensorflow/

export PYTHONPATH=":tirenv/lib/python3.5/site-packages/tensorflow/models:tirenv/lib/python3.5/site-packages/tensorflow/models/slim"

cd tirenv/lib/python3.5/site-packages/tensorflow/models

protoc object_detection/protos/*.proto --python_out=.
~~~~

then simply navigate back to where rgui is and run
~~~~
python rgui.py
~~~~

