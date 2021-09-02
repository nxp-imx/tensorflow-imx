# Release 2.5.0 for LF5.10.52-2.1.0

## Major Features and Improvements

*   `tf.lite`:
    *   Added VX Delegate to TensorFlow Lite. VX Delegate is an alternative Delegate to offload ML inference to i.MX8 on-chip accelerators -GPU or NPU.

## Known Issue and Limitation
*   Fails to build evaluation tools with Yocto SDK due to missing
    protobuf include files for TensorFlow Lite in Yocto SDK
    *   TensorFlow Lite uses a different version of protobuf than available in
        Yocto SDK (3.9.2 vs 3.15.2). The protobuf for TensorFlow Lite (tensorflow-protobuf-dev package)
        is not installed on generated Yocto SDK, therefore attempt to build 
        the TensorFlow Lite model evaluation tools fails.

        The tensorflow-protobuf-dev (libprotobuf-dev_3.9.2-r0_arm64.deb) package
        needs to manually extract into the Yocto SDK:

        ```dpkg -x  libprotobuf-dev_3.9.2-r0_arm64.deb <PATH_TO_YOCTO_SDK>/sysroots/cortexa53-crypto-poky-linux/```

        This package is located in `tmp/deploy/deb/cortexa53-crypto/` in the Yocto build folder.
*   Implicit padding for TransposeConv2D is not supported in NNAPI implementation
    *   Models using implicit padding schema for TransposeConv2D fails
        to run using NNAPI Delegate, as the underlying NNAPI implementation
        not support implicit padding schema.

        Use VX Delegate with these models.
