sudo apt install protobuf-compiler
source newenv/bin/activate
pip install tensorflow pillow opencv-python
cp -r models newenv/lib/python3.5/site-packages/tensorflow/
export PYTHONPATH=":newenv/lib/python3.5/site-packages/tensorflow/models:newenv/lib/python3.5/site-packages/tensorflow/models/slim"
cd newenv/lib/python3.5/site-packages/tensorflow/models
protoc object_detection/protos/*.proto --python_out=.
