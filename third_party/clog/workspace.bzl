"""Loads the clog library, used by cpuinfo and XNNPACK."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # Attention: TensorFlow Lite CMake build uses these variables, update only the hash contents. 
    CLOG_COMMIT = "5e63739504f0f8e18e941bd63b2d6d42536c7d90"
    CLOG_SHA256 = "18eca9bc8d9c4ce5496d0d2be9f456d55cbbb5f0639a551ce9c8bac2e84d85fe"

    tf_http_archive(
        name = "clog",
        strip_prefix = "cpuinfo-d5e37adf1406cf899d7d9ec1d317c47506ccb970",
        sha256 = CLOG_SHA256,
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/{commit}.tar.gz".format(commit = CLOG_COMMIT)),
        build_file = "//third_party/clog:clog.BUILD",
    )
