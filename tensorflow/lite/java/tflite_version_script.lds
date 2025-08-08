VERS_1.0 {
  # Export JNI and native C symbols.
  global:
    Java_*;
    TfLite*;

  # Required for libunwind. This is needed if built and then run internally.

  # Hide everything else.
  local:
    *;
};
