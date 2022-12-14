"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # Attention: TensorFlow Lite CMake build uses these variables, update only the URL and the checksum value.
    FLATBUFFERS_URL = "https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz"
    FLATBUFFERS_SHA256 = "62f2223fb9181d1d6338451375628975775f7522185266cd5296571ac152bc45"

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
