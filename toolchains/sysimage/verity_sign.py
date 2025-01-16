#!/usr/bin/env python3
#
# Runs veritysetup on a partition image to compute hash tree inside it,
# and outputs root hash to a separate file.
#
import argparse
import os
import re
import time
import subprocess
import shutil
import sys

root_hash_re = re.compile("Root hash:[ \t]+([a-f0-9]+).*")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input (tzst) file of tree to operate in", type=str)
    parser.add_argument("-o", "--output", help="Target (tzst) file of tree to write to", type=str)
    parser.add_argument("-r", "--root-hash", help="Output file containing root hash", type=str)
    parser.add_argument(
        "-s",
        "--salt",
        help="Salt to use for veritysetup",
        type=str,
        default="89bc06702975f10b41dd53e26eb9aef0adde93bc13d73e8f26b496dddefe2483",
    )
    parser.add_argument(
        "-u", "--uuid", help="UUID to use for veritysetup", type=str, default="1736291d-aceb-42d7-b044-43ee31d86cd4"
    )
    parser.add_argument(
        "-H",
        "--hash-offset",
        help="Offset where hash tree is placed in image",
        type=int,
        default=10 * 1024 * 1024 * 1024 - 128 * 1024 * 1024,
    )
    parser.add_argument("--dflate", help="Path to our dflate tool", type=str)

    args = parser.parse_args(sys.argv[1:])

    tmpdir = os.getenv("ICOS_TMPDIR")
    if not tmpdir:
        raise RuntimeError("ICOS_TMPDIR env variable not available, should be set in BUILD script.")

    # Copy input to output before processing it
    subprocess.run(["cp", "--no-preserve=mode,ownership", args.input, args.output], check=True)

    verity_cmdline = [
        "/usr/sbin/veritysetup",
        "format",
        args.output,
        args.output,
        "--hash-offset",
        str(args.hash_offset),
        "--uuid",
        args.uuid,
        "--salt",
        args.salt,
    ]

    with subprocess.Popen(verity_cmdline, stdout=subprocess.PIPE) as proc:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()

            # Try to parse the "completion" from the line received so
            # we can obtain the hash of the image.
            m = root_hash_re.match(line.decode("utf-8"))
            if m:
                root_hash = m.groups()[0]

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("Verity setup failed")

    with open(args.root_hash, "w") as f:
        f.write(root_hash + "\n")


if __name__ == "__main__":
    main()
