#!/usr/bin/env bash
# Install the legacy mujoco-py build and software-rendering dependencies locally.
# This needs no sudo: conda supplies GLEW/patchelf and apt only downloads/extracts
# the OSMesa development package into the repository-local prefix.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="$root/.mujoco-build"
deb_dir="$prefix/debs"

command -v conda >/dev/null || {
  echo "conda is required to create $prefix" >&2
  exit 1
}
command -v apt-get >/dev/null || {
  echo "apt-get is required to download the OSMesa packages" >&2
  exit 1
}
command -v dpkg-deb >/dev/null || {
  echo "dpkg-deb is required to extract the OSMesa packages" >&2
  exit 1
}

conda create --yes --prefix "$prefix" --channel conda-forge glew patchelf libegl
mkdir -p "$deb_dir"
(
  cd "$deb_dir"
  apt-get download libosmesa6-dev libosmesa6
  dpkg-deb -x libosmesa6-dev_*.deb "$prefix"
  dpkg-deb -x libosmesa6_*.deb "$prefix"
  rm -f libosmesa6-dev_*.deb libosmesa6_*.deb
)

echo "Installed user-local MuJoCo runtime at $prefix"
