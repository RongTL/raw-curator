#!/usr/bin/env bash
# Host-level bootstrap. Run ONCE on a fresh Ubuntu 26.04 host.
# Installs nvidia-container-toolkit + podman-compose, generates the CDI spec,
# and smoke-tests rootless GPU passthrough.
#
# Usage:  ssh -t desktop 'bash -s' < scripts/host-bootstrap.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo 'This script must run as root. Invoke via: ssh desktop "sudo bash -s" < scripts/host-bootstrap.sh'
  exit 1
fi

ORIG_USER=${SUDO_USER:-$USER}

# Ensure prereqs (curl, gnupg) exist before we touch repos.
apt-get update -qq
apt-get install -y curl gnupg2 ca-certificates >/dev/null

echo '== Adding NVIDIA Container Toolkit apt repo =='
install -d -m 0755 /usr/share/keyrings
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit podman-compose

echo '== Generating CDI spec =='
mkdir -p /etc/cdi
nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list

echo "== Rootless GPU smoke test as $ORIG_USER =="
if sudo -u "$ORIG_USER" podman run --rm --device nvidia.com/gpu=all \
     docker.io/nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi; then
  echo '== Rootless GPU OK =='
else
  echo 'WARN: rootless GPU failed; rootful (sudo podman) will still work.'
fi

echo 'Bootstrap complete.'
