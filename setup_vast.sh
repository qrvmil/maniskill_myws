#!/usr/bin/env bash

set -euo pipefail

echo "========================================"
echo "  Vast.ai machine bootstrap"
echo "  Codex + GitHub CLI + Hugging Face"
echo "========================================"

# ------------------------------------------------------------
# Base packages
# ------------------------------------------------------------

echo
echo "[1/5] Installing base packages..."

apt-get update

DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git \
  git-lfs \
  curl \
  wget \
  ca-certificates \
  gnupg \
  build-essential \
  pkg-config \
  python3 \
  python3-pip \
  python3-venv \
  tmux \
  htop \
  jq \
  unzip \
  rsync

git lfs install


# ------------------------------------------------------------
# Codex
# ------------------------------------------------------------

echo
echo "[2/5] Installing Codex..."

if command -v codex >/dev/null 2>&1; then
    echo "Codex already installed:"
    codex --version || true
else
    curl -fsSL https://chatgpt.com/codex/install.sh | sh
fi

# Reload possible PATH changes
source ~/.bashrc 2>/dev/null || true
hash -r

echo
echo "Codex version:"
codex --version || true


# ------------------------------------------------------------
# GitHub CLI
# ------------------------------------------------------------

echo
echo "[3/5] Installing GitHub CLI..."

if ! command -v gh >/dev/null 2>&1; then

    mkdir -p -m 755 /etc/apt/keyrings

    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      -o /etc/apt/keyrings/githubcli-archive-keyring.gpg

    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list

    apt-get update
    apt-get install -y gh

else
    echo "GitHub CLI already installed:"
    gh --version | head -1
fi


# ------------------------------------------------------------
# Hugging Face
# ------------------------------------------------------------

echo
echo "[4/5] Installing Hugging Face CLI..."

python3 -m pip install -U huggingface_hub --break-system-packages 2>/dev/null \
    || python3 -m pip install -U huggingface_hub

HF_HOME_DIR="/workspace/.cache/huggingface"

mkdir -p "$HF_HOME_DIR"

# Add HF_HOME only if it isn't there already
if ! grep -q 'HF_HOME=/workspace/.cache/huggingface' ~/.bashrc 2>/dev/null; then
    echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
fi

export HF_HOME="$HF_HOME_DIR"

echo "HF_HOME=$HF_HOME"


# ------------------------------------------------------------
# Authentication
# ------------------------------------------------------------

echo
echo "[5/5] Authentication"
echo

echo "----------------------------------------"
echo "Codex login"
echo "----------------------------------------"

if codex login status >/dev/null 2>&1; then
    echo "Codex already authenticated."
else
    codex login --device-auth
fi


echo
echo "----------------------------------------"
echo "GitHub login"
echo "----------------------------------------"

if gh auth status >/dev/null 2>&1; then
    echo "GitHub already authenticated."
else
    gh auth login \
      --hostname github.com \
      --git-protocol https \
      --web

    gh auth setup-git
fi


echo
echo "----------------------------------------"
echo "Git identity"
echo "----------------------------------------"

CURRENT_GIT_NAME="$(git config --global user.name || true)"
CURRENT_GIT_EMAIL="$(git config --global user.email || true)"

if [[ -z "$CURRENT_GIT_NAME" ]]; then
    read -rp "GitHub name: " GIT_NAME
    git config --global user.name "$GIT_NAME"
else
    echo "Git name: $CURRENT_GIT_NAME"
fi

if [[ -z "$CURRENT_GIT_EMAIL" ]]; then
    read -rp "GitHub email: " GIT_EMAIL
    git config --global user.email "$GIT_EMAIL"
else
    echo "Git email: $CURRENT_GIT_EMAIL"
fi


echo
echo "----------------------------------------"
echo "Hugging Face login"
echo "----------------------------------------"

if hf auth whoami >/dev/null 2>&1; then
    echo "Hugging Face already authenticated."
else
    hf auth login
fi


# ------------------------------------------------------------
# Final checks
# ------------------------------------------------------------

echo
echo "========================================"
echo "Final checks"
echo "========================================"

echo
echo "Codex:"
command -v codex
codex --version || true

echo
echo "GitHub:"
command -v gh
gh auth status || true

echo
echo "Hugging Face:"
command -v hf
hf auth whoami || true

echo
echo "Git:"
git --version
echo "name  = $(git config --global user.name || echo '<not set>')"
echo "email = $(git config --global user.email || echo '<not set>')"

echo
echo "HF_HOME=$HF_HOME"

echo
echo "========================================"
echo "Setup complete."
echo "========================================"
