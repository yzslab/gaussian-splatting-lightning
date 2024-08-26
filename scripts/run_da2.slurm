#!/bin/bash
if [ "${1}" == "" ] || [ "${2}" == "" ]; then
  echo "VENV name and image directory must be provided"
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate ${1}


srun python utils/run_depth_anything_v2.py ${2} --preview