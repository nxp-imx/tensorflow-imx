"""Loads Vulkan-Headers, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # Attention: TensorFlow Lite CMake build uses this variable, update only the hash content. 
    VULKAN_HEADERS_COMMIT = "32c07c0c5334aea069e518206d75e002ccd85389"

    tf_http_archive(
        name = "vulkan_headers",
        strip_prefix = "Vulkan-Headers-32c07c0c5334aea069e518206d75e002ccd85389",
        sha256 = "602aedcc4c6057473d0f7fee1bcc3aa01bf191371b2b5bbca949cebc03cf393a",
        link_files = {
            "//third_party/vulkan_headers:tensorflow/vulkan_hpp_dispatch_loader_dynamic.cc": "tensorflow/vulkan_hpp_dispatch_loader_dynamic.cc",
        },
        urls = tf_mirror_urls("https://github.com/KhronosGroup/Vulkan-Headers/archive/{commit}.tar.gz".format(commit = VULKAN_HEADERS_COMMIT)),
        build_file = "//third_party/vulkan_headers:vulkan_headers.BUILD",
    )
