pip install trimesh open3d tensorboard Cython pykdtree

cd ..
git clone https://github.com/kacperkan/light-field-distance
cd light-field-distance
python setup.py install

cd ../aro-net

pip install torch torchvision torchaudio

python setup.py build_ext --inplace

mkdir ssl
cd ssl
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes - batch
