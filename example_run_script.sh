source .venv_sglangcuda12/bin/activate

export PATH="$PWD/.venv_sglangcuda12/bin:/usr/local/bin:/usr/bin:/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOME="$PWD/.venv_sglangcuda12/lib/python3.12/site-packages/nvidia"

mkdir -p artifacts/backend-videos/sglang-fastwan-vsa

timeout 300s sglang generate \
  --model-path FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
  --attention-backend=video_sparse_attn \
  --VSA-sparsity=0.5 \
  --num-gpus=1 \
  --prompt "A calm ocean wave at sunrise" \
  --height=448 \
  --width=832 \
  --num-frames=61 \
  --num-inference-steps=3 \
  --seed=123 \
  --save-output \
  --output-path artifacts/backend-videos/sglang-fastwan-vsa \
  --output-file-name fastwan-vsa-self-test.mp4

test -s artifacts/backend-videos/sglang-fastwan-vsa/fastwan-vsa-self-test.mp4
