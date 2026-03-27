set -euo pipefail
export JOBTAG="job-${SLURM_JOB_ID}"
source ~/.bashrc

# Paths
ENROOT_IMAGE=/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/oliver_hausdoerfer/squash_images/olivertum+lerobot-lrzcluster+02.sqsh

# Container mounts
MOUNTS=""
MOUNTS+="/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/oliver_hausdoerfer/lerobot/outputs:/dss/dsshome1/0D/ge58guq2/lerobot_realWorldDRL/outputs,"
MOUNTS+="/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/oliver_hausdoerfer/lerobot/datasets/datasets:/dss/dsshome1/0D/ge58guq2/lerobot_realWorldDRL/datasets"

# Start interactive container shell
srun --pty \
     --container-image=$ENROOT_IMAGE \
     --container-mounts=$MOUNTS \
     bash