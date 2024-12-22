export PATH="/opt/anaconda3/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64
export CUDA_VISIBLE_DEVICES=3
source activate robert_venv
echo "current value CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

