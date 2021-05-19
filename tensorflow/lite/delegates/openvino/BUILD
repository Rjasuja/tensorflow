load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "internal_visibility_allowlist", "tflite_portable_test_suite_combined")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

EMSCRIPTEN_LINKOPTS = [
    "-s ASSERTIONS=2",
    "-s ERROR_ON_UNDEFINED_SYMBOLS=1",
    "-s DEMANGLE_SUPPORT=1",
    "-s EXIT_RUNTIME=1",
    "-s ALLOW_MEMORY_GROWTH=1",
    "-s TOTAL_MEMORY=134217728",
]

exports_files([
    "openvino_delegate.h",
])

cc_library(
    name = "openvino_delegate",
    srcs = ["openvino_delegate.cc"],
    hdrs = ["openvino_delegate.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "@intel_openvino//:openvino",
    ],
)

cc_binary(
    name = "openvino_external_delegate.so",
    srcs = [
        "openvino_delegate_adapter.cc",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":openvino_delegate",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
    ],
)

cc_library(
    name = "openvino_delegate_hdrs_only",
    hdrs = ["openvino_delegate.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility_allowlist(),
    deps = [
        "//tensorflow/lite/c:common",
    ],
)

cc_library(
    name = "openvino_delegate_test_mode",
    srcs = ["openvino_delegate.cc"],
    hdrs = ["openvino_delegate.h"],
    copts = tflite_copts() + ["-DOPENVINO_DELEGATE_TEST_MODE=1"],
    linkstatic = True,
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "@intel_openvino//:openvino",
    ],
)
