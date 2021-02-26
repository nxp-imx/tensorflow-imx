"""Loads VeriSilicon TIM-VX, used by TF Lite."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "tim_vx",
        strip_prefix = "TIM-VX-1.1.30.3",
        sha256 = "2c931684658d68fc51853f3d6ccad05b672f67f03b5c75bb634fbd88e9a568ee",
        urls = [
            "https://github.com/VeriSilicon/TIM-VX/archive/refs/tags/v1.1.30.3.tar.gz",
        ],
    )

    # Uncomment for local development
    #
    # native.local_repository (
    #     name = "tim_vx",
    #     path = "tensorflow/lite/delegates/vx-delegate/tim-vx",
    # )

    http_archive(
        name = "aarch64_A311D",
        sha256 = "9c3fe033f6d012010c92ed1f173b5410019ec144ddf68cbc49eaada2b4737e7f",
        strip_prefix = "aarch64_A311D_D312513_A294074_R311680_T312233_O312045",
        urls = [
            "https://github.com/VeriSilicon/TIM-VX/releases/download/v1.1.28/aarch64_A311D_D312513_A294074_R311680_T312233_O312045.tgz",
        ],
    )

