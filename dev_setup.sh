pip install trimesh open3d tensorboard Cython

cd ..
git clone https://github.com/kacperkan/light-field-distance
cd light-field-distance
python setup.py install

cd ../aro-net

pip install torch torchvision torchaudio

python setup.py build_ext --inplace
