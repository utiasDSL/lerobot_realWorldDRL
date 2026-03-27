#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8,mcml-hgx-a100-80x4,mcml-hgx-h100-94x4
#SBATCH -q mcml
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH -o /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/oliver_hausdoerfer/slurm_logs/output_%A_%a.log
#SBATCH -e /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/oliver_hausdoerfer/slurm_logs/error_%A_%a.log
#SBATCH --mail-user=oliver.hausdoerfer@tum.de

# Source your .bashrc file to get environment variables
set  -euo  pipefail
export  JOBTAG="job-${SLURM_JOB_ID}"
source ~/.bashrc

# Paths
ENROOT_IMAGE=/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/oliver_hausdoerfer/squash_images/olivertum+lerobot-lrzcluster+02.sqsh

# Container mounts
MOUNTS+="/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/oliver_hausdoerfer/lerobot/outputs:/dss/dsshome1/0D/ge58guq2/lerobot/outputs" # redirect logging to project storage
MOUNTS+="/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/oliver_hausdoerfer/lerobot/datasets/datasets:/dss/dsshome1/0D/ge58guq2/lerobot/datasets" # mount downloaded datasets

srun --container-image=$ENROOT_IMAGE \
     --container-mounts=$MOUNTS \
     --container-workdir=/dss/dsshome1/0D/ge58guq2/lerobot \
     bash -c "lerobot-train --config_path=config/smolvla/smolvla.yaml"

