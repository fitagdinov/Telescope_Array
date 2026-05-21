#export PATH="/opt/anaconda3/bin:$PATH"
#export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64
#export CUDA_VISIBLE_DEVICES=1
#source activate robert_venv
#echo "current value CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# Override PATH to prioritize Miniconda
#export PATH="/home/rfit/miniconda3/envs/python312-env/bin:$PATH"
export PATH="/home/rfit/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64
export CUDA_VISIBLE_DEVICES=0
#source activate robert_venv
#source /home/rfit/miniconda3/bin/activate python312-env
source /home/rfit/miniconda3/bin/activate python39
#conda activate base
echo $(python --version)
echo "current value CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
