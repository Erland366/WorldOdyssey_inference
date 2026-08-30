#!/usr/bin/env bash

worldodyssey_prepend_path() {
    local var_name="$1"
    local entry="$2"
    local current_value="${!var_name:-}"

    case ":$current_value:" in
        *":$entry:"*) ;;
        *)
            if [[ -n "$current_value" ]]; then
                export "$var_name=$entry:$current_value"
            else
                export "$var_name=$entry"
            fi
            ;;
    esac
}

worldodyssey_configure_minimax_h3_env() {
    local venv_path="$1"
    local project_root="${2:-$(pwd)}"
    local site_packages
    local lib_dir
    local found_nvidia_lib=0
    local hf_home

    worldodyssey_prepend_path PATH "$venv_path/bin"

    hf_home="${WORLDODYSSEY_HF_HOME:-${HF_HOME:-$project_root/.cache/huggingface}}"
    export HF_HOME="$hf_home"
    export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
    export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
    mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

    for site_packages in "$venv_path"/lib/python*/site-packages; do
        [[ -d "$site_packages" ]] || continue
        for lib_dir in "$site_packages"/nvidia/*/lib "$site_packages"/nvidia/cu13/lib; do
            [[ -d "$lib_dir" ]] || continue
            worldodyssey_prepend_path LD_LIBRARY_PATH "$lib_dir"
            found_nvidia_lib=1
        done
    done

    if [[ "$found_nvidia_lib" -eq 0 ]]; then
        echo "MiniMax-H3 NVIDIA libraries were not found under $venv_path." >&2
        echo "Run scripts/install_minimax_h3.sh before starting the H3 server." >&2
        return 1
    fi
}
