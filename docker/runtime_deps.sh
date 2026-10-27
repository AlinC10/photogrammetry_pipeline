#!/bin/sh
# Collects the shared libraries the given binaries need (resolved with ldd in the BUILD stage) into DEST,
# so the runtime stage can start from nvidia/cuda:*-base instead of installing -dev packages.
# Skipped: glibc / libstdc++ / libgcc (already in the base image) and the NVIDIA driver libraries
# (libcuda, libnvidia-*), which the NVIDIA container runtime mounts from the host at run time.
#
#   docker_runtime_deps.sh /opt/runtime-libs /opt/openmvs_bin/bin/OpenMVS/* /opt/mvs-texturing/.../texrecon
set -eu
DEST="$1"; shift
mkdir -p "$DEST"
for bin in "$@"; do
    [ -f "$bin" ] || continue
    ldd "$bin" 2>/dev/null | awk '/=> \// {print $3}' | while read -r lib; do
        case "$(basename "$lib")" in
            libc.so*|libm.so*|libpthread.so*|libdl.so*|librt.so*|ld-linux*|libgcc_s.so*|libstdc++.so*) continue ;;
            libcuda.so*|libnvidia-*) continue ;;
        esac
        # copy the real file under the soname the binary asks for
        [ -e "$DEST/$(basename "$lib")" ] || cp -L "$lib" "$DEST/$(basename "$lib")"
    done
done
echo "Collected $(ls "$DEST" | wc -l) libraries ($(du -sh "$DEST" | cut -f1)) into $DEST"
