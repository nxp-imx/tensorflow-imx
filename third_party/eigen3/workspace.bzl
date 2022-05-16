"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Eigen."""

    # Attention: TensorFlow Lite CMake build uses this variable, update only the hash content.
    EIGEN_COMMIT = "085c2fc5d53f391afcccce21c45e15f61c827ab1"
    EIGEN_SHA256 = "cd72f0a56a95d85cb8a0160f4adc7fea72da49fbb7351ebb31c4e67e1a5fc8bd"

    tf_http_archive(
        name = "eigen_archive",
        build_file = "//third_party/eigen3:eigen_archive.BUILD",
        sha256 = EIGEN_SHA256,
        strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
        urls = tf_mirror_urls("https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT)),
    )
