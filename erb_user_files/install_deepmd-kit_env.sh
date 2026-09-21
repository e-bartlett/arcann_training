#!/usr/bin/env bash
#
# install_deepmd-kit_env.sh
#
# Reproduces the deepmd-kit environment used for this ArcaNN fork
# (/kuhpc/work/thompson/e497b540/.conda/deepmd-kit) on your own account.
#
# IMPORTANT: this is NOT a hand-built `conda create -n ...` environment.
# It's DeePMD-kit's official "offline packages" installer -- a
# self-contained conda distribution that ships with deepmd-kit, TensorFlow,
# PyTorch, and a LAMMPS build with the DeePMD plugin already compiled in,
# with matching CUDA/cuDNN versions baked in. Building that stack from
# source or from conda-forge yourself is a multi-hour headache; this is
# the reliable way to get the exact same thing. Details:
#   https://docs.deepmodeling.com/projects/deepmd/en/latest/install/easy-install-dev.html
#
# The reference install on this cluster is deepmd-kit v3.0.1, cuda126
# variant (~10 GB unpacked). This script reproduces that by default; pass
# -v/-t to grab a different version or variant (cpu, cuda118, ...) -- see
# what's available for a given version at:
#   https://github.com/deepmodeling/deepmd-kit/releases
#
# Usage:
#   ./install_deepmd-kit_env.sh [-p PREFIX] [-v VERSION] [-t VARIANT]
#
#   -p PREFIX   Install location (default: $HOME/.conda/deepmd-kit)
#   -v VERSION  deepmd-kit release version (default: 3.0.1)
#   -t VARIANT  cpu | cuda126 | cuda118 | ... (default: cuda126)
#
# Requires: bash, curl, python3, and (for the cuda126 default) ~15 GB of
# free disk space for the download plus the ~10 GB unpacked environment.
#
# After it finishes, activate the environment with:
#   source "$PREFIX/etc/profile.d/conda.sh"
#   conda activate "$PREFIX"

set -euo pipefail

PREFIX="${HOME}/.conda/deepmd-kit"
VERSION="3.0.1"
VARIANT="cuda126"

while getopts "p:v:t:h" opt; do
  case "$opt" in
    p) PREFIX="$OPTARG" ;;
    v) VERSION="$OPTARG" ;;
    t) VARIANT="$OPTARG" ;;
    h) grep '^#' "$0" | sed 's/^#//'; exit 0 ;;
    *) exit 1 ;;
  esac
done

if [[ -e "$PREFIX" ]]; then
  echo "Error: $PREFIX already exists. Pick a different -p PREFIX." >&2
  exit 1
fi

REPO="deepmodeling/deepmd-kit"
TAG="v${VERSION}"
ASSET_PREFIX="deepmd-kit-${VERSION}-${VARIANT}-Linux-x86_64.sh"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
cd "$WORKDIR"

echo "==> Looking up release ${TAG} assets for variant '${VARIANT}'..."
curl -fsSL "https://api.github.com/repos/${REPO}/releases/tags/${TAG}" -o release.json

# Installers over ~2 GB get split by GitHub into numbered parts (e.g.
# name.sh.0/.1 on older releases, name.sh.00/.01/.02 on newer ones). Pull
# whatever parts exist for this variant, in name order, then reassemble.
mapfile -t PARTS < <(python3 - "$ASSET_PREFIX" release.json <<'PYEOF'
import json, sys
prefix, path = sys.argv[1], sys.argv[2]
data = json.load(open(path))
names = sorted(
    a["name"] for a in data["assets"]
    if a["name"] == prefix or a["name"].startswith(prefix + ".")
)
for n in names:
    print(n)
PYEOF
)

if [[ ${#PARTS[@]} -eq 0 ]]; then
  echo "Error: no assets found matching '${ASSET_PREFIX}' in release ${TAG}." >&2
  echo "Check available versions/variants at: https://github.com/${REPO}/releases" >&2
  exit 1
fi

echo "==> Downloading ${#PARTS[@]} part(s):"
for part in "${PARTS[@]}"; do
  echo "    $part"
  curl -fL --retry 3 -o "$part" \
    "https://github.com/${REPO}/releases/download/${TAG}/${part}"
done

INSTALLER="${ASSET_PREFIX}"
if [[ ${#PARTS[@]} -gt 1 ]]; then
  echo "==> Reassembling split installer..."
  cat "${PARTS[@]}" > "$INSTALLER"
fi
chmod +x "$INSTALLER"

echo "==> Installing to ${PREFIX} (unpacking TensorFlow/PyTorch/LAMMPS -- a few minutes)..."
mkdir -p "$(dirname "$PREFIX")"
bash "$INSTALLER" -b -p "$PREFIX"

cat <<EOF

==> Done. To use this environment:

    source "${PREFIX}/etc/profile.d/conda.sh"
    conda activate "${PREFIX}"

That puts dp, lmp (LAMMPS with the DeePMD plugin), python, etc. on PATH.
Add those two lines to your job scripts (see erb_user_files/job_*_kuhpc.sh
for examples) or your ~/.bashrc as needed.
EOF
