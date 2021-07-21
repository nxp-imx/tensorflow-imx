# VX-delegate
vx-delegate constructed with TIM-VX as an openvx delegate for tensorflow lite.

# build from source

```sh
# clone TIM-VX
cd tensorflow/lite/delegates/vx-delegate
git clone https://github.com/VeriSilicon/TIM-VX.git tim-vx

# Modify WORKSPACE
modify WORKSPACE file to use new_local_repository instead of default
http_archive

# build
bazel build tensorflow/lite/delegates/vx-delegate:vx_delegate
bazel build tensorflow/lite/kernels:conv_test
```
