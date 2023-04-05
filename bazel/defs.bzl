"""
Utilities for building IC replica and canisters.
"""

load("@rules_rust//rust:defs.bzl", "rust_binary", "rust_test")

_COMPRESS_CONCURENCY = 16

def _compress_resources(_os, _input_size):
    """ The function returns resource hints to bazel so it can properly schedule actions.

    Check https://bazel.build/rules/lib/actions#run for `resource_set` parameter to find documentation of the function, possible arguments and expected return value.
    """
    return {"cpu": _COMPRESS_CONCURENCY}

def _gzip_compress(ctx):
    """GZip-compresses source files.
    """
    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.run_shell(
        command = "{pigz} --processes {concurency} --no-name {srcs} --stdout > {out}".format(pigz = ctx.file._pigz.path, concurency = _COMPRESS_CONCURENCY, srcs = " ".join([s.path for s in ctx.files.srcs]), out = out.path),
        inputs = ctx.files.srcs,
        outputs = [out],
        tools = [ctx.file._pigz],
        resource_set = _compress_resources,
    )
    return [DefaultInfo(files = depset([out]), runfiles = ctx.runfiles(files = [out]))]

gzip_compress = rule(
    implementation = _gzip_compress,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "_pigz": attr.label(allow_single_file = True, default = "@pigz"),
    },
)

def _zstd_compress(ctx):
    """zstd-compresses source files.
    """
    out = ctx.actions.declare_file(ctx.label.name)

    # TODO: install zstd as depedency.
    ctx.actions.run(
        executable = "zstd",
        arguments = ["--threads=0", "-10", "-f", "-z", "-o", out.path] + [s.path for s in ctx.files.srcs],
        inputs = ctx.files.srcs,
        outputs = [out],
        env = {"ZSTDMT_NBWORKERS_MAX": str(_COMPRESS_CONCURENCY)},
        resource_set = _compress_resources,
    )
    return [DefaultInfo(files = depset([out]), runfiles = ctx.runfiles(files = [out]))]

zstd_compress = rule(
    implementation = _zstd_compress,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
    },
)

def rust_test_suite_with_extra_srcs(name, srcs, extra_srcs, **kwargs):
    """ A rule for creating a test suite for a set of `rust_test` targets.

    Like `rust_test_suite`, but with ability to deal with integration
    tests that use common utils across various tests.  The sources of
    the common utils should be specified in extra_srcs` argument.

    Args:
      name: see description for `rust_test_suite`
      srcs: see description for `rust_test_suite`
      extra_srcs: list of files that e.g. implement common utils, must be disjoint from `srcs`
      **kwargs: see description for `rust_test_suite`
    """
    tests = []

    for extra_src in extra_srcs:
        if not extra_src.endswith(".rs"):
            fail("Wrong file in extra_srcs: " + extra_src + ". extra_srcs should have `.rs` extensions")

    for src in srcs:
        if not src.endswith(".rs"):
            fail("Wrong file in srcs: " + src + ". srcs should have `.rs` extensions")

        # Prefixed with `name` to allow parameterization with macros
        # The test name should not end with `.rs`
        test_name = name + "_" + src[:-3]
        rust_test(
            name = test_name,
            srcs = [src] + extra_srcs,
            crate_root = src,
            **kwargs
        )
        tests.append(test_name)

    native.test_suite(
        name = name,
        tests = tests,
        tags = kwargs.get("tags", None),
    )

def rust_bench(name, env = {}, data = [], **kwargs):
    """A rule for defining a rust benchmark.

    Args:
      name: the name of the executable target.
      env: additional environment variables to pass to the benchmark binary.
      data: data dependencies required to run the benchmark.
      **kwargs: see docs for `rust_binary`.
    """
    binary_name = "_" + name + "_bin"
    rust_binary(name = binary_name, **kwargs)
    native.sh_binary(
        srcs = ["//bazel:generic_rust_bench.sh"],
        # Allow benchmark targets to use test-only libraries.
        name = name,
        testonly = kwargs.get("testonly", False),
        env = dict(env.items() + {"BAZEL_DEFS_BENCH_BIN": "$(location :%s)" % binary_name}.items()),
        data = data + [":" + binary_name],
        tags = kwargs.get("tags", []) + ["rust_bench"],
    )
