"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # Attention: TensorFlow Lite CMake build uses these variables, update only the URL and the checksum value.
    FLATBUFFERS_URL = "https://github.com/google/flatbuffers/archive/v23.5.26.tar.gz"
    FLATBUFFERS_SHA256 = "1cce06b17cddd896b6d73cc047e36a254fb8df4d7ea18a46acf16c4c0cd3f3f3"

    tf_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-23.5.26",
        sha256 = FLATBUFFERS_SHA256,
        urls = tf_mirror_urls(FLATBUFFERS_URL),
        build_file = "//third_party/flatbuffers:flatbuffers.BUILD",
        system_build_file = "//third_party/flatbuffers:BUILD.system",
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )
