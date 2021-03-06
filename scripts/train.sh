set -ex
CLASS='furniture'  # facades, day2night, edges2shoes, edges2handbags, maps
MODEL='bicycle_gan'
GPU_ID=6
NZ=8
COMMENT='lz-10times-l12times-lr20%-learn30e-decay25e'
lambda_L1=10
lambda_z=0.5
BATCH=32

CHECKPOINTS_DIR=./checkpoints/${CLASS}/
DATE=`date '+%m_%d_%H_%M_%S'`
NAME=${CLASS}_${MODEL}_${DATE}
DISPLAY_ID=$((GPU_ID*10+1))
PORT=8097

# dataset
NO_FLIP=''
DIRECTION='AtoB'
LOAD_SIZE=286
CROP_SIZE=256
INPUT_NC=3
LR=0.0002


# dataset parameters
case ${CLASS} in
'facades')
  NITER=200
  NITER_DECAY=200
  SAVE_EPOCH=25
  DIRECTION='BtoA'
  ;;
'edges2shoes')
  NITER=30
  NITER_DECAY=30
  LOAD_SIZE=256
  SAVE_EPOCH=5
  INPUT_NC=1
  NO_FLIP='--no_flip'
  ;;
'edges2handbags')
  NITER=15
  NITER_DECAY=15
  LOAD_SIZE=256
  SAVE_EPOCH=5
  INPUT_NC=1
  ;;
'maps')
  NITER=200
  NITER_DECAY=200
  LOAD_SIZE=600
  SAVE_EPOCH=25
  DIRECTION='BtoA'
  ;;
'furniture')
  NITER=30
  NITER_DECAY=25
  SAVE_EPOCH=100
  LOAD_SIZE=256
  NZ=4
  lambda_L1=10
  lambda_z=5
  LR=0.00004
  ;;
'day2night')
  NITER=50
  NITER_DECAY=50
  SAVE_EPOCH=10
  ;;
*)
  echo 'WRONG category'${CLASS}
  ;;
esac



# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./train.py \
  --display_id ${DISPLAY_ID} \
  --dataroot ./datasets/${CLASS} \
  --name ${NAME} \
  --model ${MODEL} \
  --display_port ${PORT} \
  --direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --nz ${NZ} \
  --save_epoch_freq ${SAVE_EPOCH} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --comment ${COMMENT} \
  --use_dropout \
  --lambda_L1 ${lambda_L1} \
  --lambda_z ${lambda_z} \
  --batch_size ${BATCH} \
  --display_env ${NAME} \
  --lr ${LR}