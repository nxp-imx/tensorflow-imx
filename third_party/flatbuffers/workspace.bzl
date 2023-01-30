"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # Attention: TensorFlow Lite CMake build uses these variables, update only the URL and the checksum value.
    FLATBUFFERS_URL = "https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz"
    FLATBUFFERS_SHA256 = "01a2c46a064601795c549fb3012530de33697a4b00f1ee88e538af08f2f11009"

    tf_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-1.12.0",
        sha256 = FLATBUFFERS_SHA256,
        urls = tf_mirror_urls(FLATBUFFERS_URL),
        build_file = "//third_party/flatbuffers:flatbuffers.BUILD",
        system_build_file = "//third_party/flatbuffers:BUILD.system",
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )
