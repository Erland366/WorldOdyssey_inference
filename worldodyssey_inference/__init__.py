import warnings

warnings.filterwarnings(
    "ignore",
    message=".*CUDA initialization: The NVIDIA driver on your system is too old.*",
    category=UserWarning,
    module="torch.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*The `local_dir_use_symlinks` argument is deprecated and ignored in `hf_hub_download`.*",
    category=UserWarning,
    module="huggingface_hub.*"
)
