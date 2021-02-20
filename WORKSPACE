workspace(name = "org_tensorflow")

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and tf_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.
load("@//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

local_repository (
    name = "tim_vx",
    path = "tensorflow/lite/delegates/vx-delegate/tim-vx",
)

http_archive(
    name = "aarch64_A311D",
    sha256 = "9c3fe033f6d012010c92ed1f173b5410019ec144ddf68cbc49eaada2b4737e7f",
    strip_prefix = "aarch64_A311D_D312513_A294074_R311680_T312233_O312045",
    urls = [
        "https://github.com/VeriSilicon/TIM-VX/releases/download/v1.1.28/aarch64_A311D_D312513_A294074_R311680_T312233_O312045.tgz",
    ],
)

#
# For developers: Enable below for development build
#
#new_local_repository (
#    name = "tim_vx",
#    path = "tensorflow/lite/delegates/vx-delegate/tim-vx",
#    build_file = "@//:tensorflow/lite/delegates/vx-delegate/tim-vx/BUILD.bazel",
#)



