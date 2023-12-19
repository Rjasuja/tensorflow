load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "internal_visibility_allowlist", "tflite_portable_test_suite_combined")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files([
    "openvino_delegate.h",
])

cc_library(
    name ="openvino_graph_builder",
    srcs = ["openvino_graph_builder.cc"],
    hdrs = ["openvino_graph_builder.h"],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/delegates/openvino/operations:operations_base",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/tools:logging",
        "@intel_openvino//:openvino",
    ],
)

cc_library(
    name ="openvino_delegate_core",
    srcs = ["openvino_delegate_core.cc"],
    hdrs = ["openvino_delegate_core.h"],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        ":openvino_graph_builder",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels:kernel_util",
        "@intel_openvino//:openvino",
    ],
)

cc_library(
    name ="openvino_delegate_kernel",
    srcs = ["openvino_delegate_kernel.cc"],
    hdrs = ["openvino_delegate_kernel.h"],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        ":openvino_delegate_core",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/kernels:kernel_util",
        "@intel_openvino//:openvino",
    ],
)

cc_library(
    name = "openvino_delegate",
    srcs = ["openvino_delegate.cc"],
    hdrs = ["openvino_delegate.h"],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        ":openvino_delegate_kernel",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "@intel_openvino//:openvino",
    ],
)

'''cc_library(
    name ="openvino_delegate_kernel",
    srcs = ["openvino_delegate_kernel.cc"],
    hdrs = ["openvino_delegate_kernel.h"],
    tags = [
        "manual",
        "nobuilder",
    ],
    linkstatic = True,
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/kernels:kernel_util",
        "@intel_openvino//:openvino",
    ],
)
'''


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
        ":openvino_delegate_kernel",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "@intel_openvino//:openvino",
    ],
)
'''
cc_test(
    name = "openvino_graph_builder_test",
    srcs = ["openvino_graph_builder_test.cc"],
    linkopts = select({
        "//conditions:default": [],
    }),
    deps = [
        ":openvino_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)
'''
