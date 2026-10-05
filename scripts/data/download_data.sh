#!/usr/bin/env bash
# Clone the GitHub-hosted sources into data/raw/ at the commits used for the paper.
# HuggingFace datasets are fetched by the builders (revisions pinned in
# src/data/sources.py); HackAPrompt is cached by scripts/data/download_hf.py.

set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"

clone_at() {
    local repo_url="$1" target="$2" commit="$3"
    if [ ! -d "$target/.git" ]; then
        echo "[clone] $repo_url -> $target"
        git clone --quiet "$repo_url" "$target"
    fi
    git -C "$target" -c advice.detachedHead=false checkout --quiet "$commit"
    echo "[ok] $target @ ${commit:0:10}"
}

clone_at https://github.com/microsoft/BIPIA                 "$RAW_DIR/BIPIA"                 a004b69ec0dd446e0afd461d98cb5e96e120a5d0
clone_at https://github.com/uiuc-kang-lab/InjecAgent        "$RAW_DIR/InjecAgent"            f19c9f2c79a41046eb13c03c51a24c567a8ffa07
clone_at https://github.com/liu00222/Open-Prompt-Injection  "$RAW_DIR/Open-Prompt-Injection" 95290f7ce3794c4c52ad3fe8113db2bfcdfe89e0
clone_at https://github.com/ethz-spylab/agentdojo           "$RAW_DIR/agentdojo"             089ed468cf3ed0322acc66b0211f26d9d90dbf60
