echo "Create virtual environment and uv sync" 

uv sync

echo "Activate virtual environment"

source .venv/bin/activate

echo "Install fastvideo on top of uv"

uv pip install fastvideo