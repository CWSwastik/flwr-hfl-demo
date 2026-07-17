#!/bin/bash
#SBATCH -p gpu_rtx_pro_6000_6_csis_hyd
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8 
#SBATCH --mem 200G
#SBATCH -t 2-00:00 # time (D-HH:MM)
#SBATCH --job-name="flwr_6000"
#SBATCH -o /home/manik/harsha/res_flwr/out_flwr_6000_.%j.txt
#SBATCH -e /home/manik/harsha/res_flwr/err_flwr_6000_.%j.txt
#SBATCH --gres=gpu:1
#SBATCH --mail-user=p20200437@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=ALL
nvidia-smi

spack load anaconda3@2022.05
conda init bash
eval "$(conda shell.bash hook)"
echo "anaconda loaded"

spack load cuda@11.6
spack load cudnn@8.4.0.27-11.6
echo "cuda loaded"

conda env list

conda activate flwr

cd /home/manik/harsha/flwr-hfl-demo

# Parallel-safe: every job remaps all topology ports to free ones,
# and run_experiments.py scopes its process cleanup to this job.
export FL_AUTO_PORTS=1

# Usage: sbatch 6000_flwr.sh <experiments.json>   (defaults to experiments.json)
echo "Running experiments: ${1:-experiments.json}"
srun python -u run_experiments.py "${1:-experiments.json}"

