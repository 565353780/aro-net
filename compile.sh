#rm -rf build

mkdir build
cd build
cmake \
  -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') \
  -DOpen3D_ROOT=../../open3d-manage/3rd/open3d-devel-linux/ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  ..
make -j
