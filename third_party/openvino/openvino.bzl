OPENVINO_NATIVE_DIR = "OPENVINO_NATIVE_DIR"
_ENVIRONS = [OPENVINO_NATIVE_DIR]

def _openvino_native_impl(repository_ctx):
    print ("*************************\n")
    print (repository_ctx.os.environ)
    print ("*************************\n")
    openvino_native_dir = repository_ctx.os.environ.get(OPENVINO_NATIVE_DIR)
    repository_ctx.symlink(openvino_native_dir, "openvino")
    repository_ctx.file("BUILD", """
cc_library(
    name = "openvino",
    hdrs = glob(["openvino/include/ie"]),
    srcs = ["openvino/lib64/libopenvino.so",
            "openvino/lib64/libopenvino.so.2023.0.2"],
    includes = ["openvino/runtime/include/ie/cpp",
                "openvino/runtime/include/ie",
                "openvino/runtime/include"],
    visibility = ["//visibility:public"],
)
    """)

openvino_configure = repository_rule(
    implementation = _openvino_native_impl,
    environ = _ENVIRONS,
)
