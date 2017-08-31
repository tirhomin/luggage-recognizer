python3.6 -m venv newenv
source newenv/bin/activate
pip install tensorflow pillow
cp -r models newenv/lib/python3.6/site-packages/tensorflow/
export PYTHONPATH=":newenv/lib/python3.6/site-packages/tensorflow/models:newenv/lib/python3.6/site-packages/tensorflow/models/slim"
cd newenv/lib/python3.6/site-packages/tensorflow/models
protoc object_detection/protos/*.proto --python_out=.
cd ../../../../../../
brew install opencv3 --with-contrib --with-python3 --without-python --with-ffmpeg
ln -s /usr/local/lib/python3.6/site-packages/cv2.cpython-36m-darwin.so newenv/lib/python3.6/site-packages/cv2.so