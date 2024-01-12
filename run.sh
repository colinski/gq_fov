#module load apptainer/1.1.5+py3.8.12
SIF=$WORK/sif/dataproc.sif
SRC_PATH=$WORK/src/gq_fov
#DATA_ROOT=/project/pi_marlin_umass_edu/2022-09-01/data
#DST=/project/pi_marlin_umass_edu/mocap_processed
#NODE=$1

singularity run --nv -H $SRC_PATH --bind /project:/project --bind /work:/work $SIF python "$@"
