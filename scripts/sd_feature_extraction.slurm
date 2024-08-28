#!/bin/bash

# 4 nodes, 2 GPUs per node: sbatch --nodes=4 --gres=gpu:2 --ntasks-per-node=2 scripts/run_da2.slurm DATASET_DIR --preview

if [ "${1}" == "" ]; then
  echo "image directory is required"
  exit 1
fi

srun python utils/sd_feature_extraction.py "${1}" "${@:2}"