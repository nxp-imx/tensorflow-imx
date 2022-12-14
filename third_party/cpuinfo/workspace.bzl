"""Loads the cpuinfo library, used by XNNPACK."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # Attention: TensorFlow Lite CMake build uses this variable, update only the hash content. 
    CPUINFO_COMMIT = "5916273f79a21551890fd3d56fc5375a78d1598d"

    tf_http_archive(
        name = "cpuinfo",
        strip_prefix = "cpuinfo-5916273f79a21551890fd3d56fc5375a78d1598d",
        sha256 = "2a160c527d3c58085ce260f34f9e2b161adc009b34186a2baf24e74376e89e6d",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/{commit}.zip".format(commit = CPUINFO_COMMIT)),
        build_file = "//third_party/cpuinfo:cpuinfo.BUILD",
    )
