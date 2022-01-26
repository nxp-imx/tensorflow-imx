"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # Attention: TensorFlow Lite CMake build uses this variable, update only the hash content. 
    RUY_COMMIT = "841ea4172ba904fe3536789497f9565f2ef64129"

    tf_http_archive(
        name = "ruy",
        sha256 = "dd6bf40322303cf8982f340e4139397c8fa350ff691d5254599cb21e0138fc65",
        strip_prefix = "ruy-841ea4172ba904fe3536789497f9565f2ef64129",
        urls = tf_mirror_urls("https://github.com/google/ruy/archive/{commit}.zip".format(commit = RUY_COMMIT)),
        build_file = "//third_party/ruy:BUILD",
    )
