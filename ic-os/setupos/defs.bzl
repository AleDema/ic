"""
Hold manifest common to all SetupOS variants.
"""

load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("//toolchains/sysimage:toolchain.bzl", "ext4_image")
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

# Declare the dependencies that we will have for the built filesystem images.
# This needs to be done separately from the build rules because we want to
# compute the hash over all inputs going into the image and derive the
# "version.txt" file from it.

def image_deps(mode, _malicious = False):
    """
    Define all SetupOS inputs.

    Args:
      mode: Variant to be built, dev or prod.
      _malicious: Unused, but currently needed to fit generic build structure.
    Returns:
      A dict containing inputs to build this image.
    """

    deps = {
        # Define rootfs and bootfs
        "bootfs": {
            # base layer
            ":rootfs-tree.tar": "/",
        },
        "rootfs": {
            # base layer
            ":rootfs-tree.tar": "/",
        },

        # Set various configuration values
        "base_image": Label("//ic-os/setupos:rootfs/docker-base." + mode),
        "docker_context": Label("//ic-os/setupos:rootfs-files"),
        "partition_table": Label("//ic-os/setupos:partitions.csv"),
        "rootfs_size": "1750M",
        "bootfs_size": "100M",
        "grub_config": Label("//ic-os/setupos:grub.cfg"),
        "extra_boot_args": Label("//ic-os/setupos:rootfs/extra_boot_args"),

        # Add any custom partitions to the manifest
        "custom_partitions": lambda: (_custom_partitions)(mode),
    }

    return deps

# Inject a step building a data partition that contains either dev or prod
# child images, depending on this build variant.
def _custom_partitions(mode):
    if mode == "dev":
        guest_image = Label("//ic-os/guestos/envs/dev:disk-img.tar.gz")
        host_image = Label("//ic-os/hostos/envs/dev:disk-img.tar.gz")
    else:
        guest_image = Label("//ic-os/guestos/envs/prod:disk-img.tar.gz")
        host_image = Label("//ic-os/hostos/envs/prod:disk-img.tar.gz")

    copy_file(
        name = "copy_guestos_img",
        src = guest_image,
        out = "guest-os.img.tar.gz",
        allow_symlink = True,
    )

    copy_file(
        name = "copy_hostos_img",
        src = host_image,
        out = "host-os.img.tar.gz",
        allow_symlink = True,
    )

    pkg_tar(
        name = "data_tar",
        srcs = [
            Label("//ic-os/setupos:data/nns_public_key.pem"),
            Label("//ic-os/setupos:deployment.json"),
            ":guest-os.img.tar.gz",
            ":host-os.img.tar.gz",
        ],
        mode = "0644",
        package_dir = "data",
    )

    ext4_image(
        name = "partition-data.tar",
        src = "data_tar",
        partition_size = "1750M",
        subdir = "./data",
        target_compatible_with = [
            "@platforms//os:linux",
        ],
    )

    return [
        Label("//ic-os/setupos:partition-config.tar"),
        ":partition-data.tar",
    ]
