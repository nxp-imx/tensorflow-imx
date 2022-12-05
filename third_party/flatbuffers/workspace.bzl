"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # Attention: TensorFlow Lite CMake build uses these variables, update only the URL and the checksum value.
    FLATBUFFERS_URL = "https://github.com/google/flatbuffers/archive/v2.0.7.tar.gz"
    FLATBUFFERS_SHA256 = "4c7986174dc3941220bf14feaacaad409c3e1526d9ad7f490366fede9a6f43fa"

    tf_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-2.0.7",
        sha256 = FLATBUFFERS_SHA256,
        urls = tf_mirror_urls(FLATBUFFERS_URL),
        build_file = "//third_party/flatbuffers:flatbuffers.BUILD",
        system_build_file = "//third_party/flatbuffers:BUILD.system",
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )
