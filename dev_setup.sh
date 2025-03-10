cd ..
git clone https://github.com/kacperkan/light-field-distance

pip install -U trimesh open3d tensorboard Cython pykdtree gradio

cd light-field-distance
python setup.py install

cd ../aro-net

pip install -U torch torchvision torchaudio

mkdir ssl
cd ssl
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes - batch
