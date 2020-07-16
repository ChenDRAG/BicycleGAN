set -ex
# models
NAME='furniture_bicycle_gan_07_15_22_59_36'
checkpoints_dir='./checkpoints/furniture/'
RESULTS_DIR=${checkpoints_dir}${NAME}/test_results
#   (opt.checkpoints_dir, opt.name), '%s_net_%s.pth' % (epoch, name)   

# dataset
CLASS='furniture'
DIRECTION='AtoB' # from domain A to domain B
LOAD_SIZE=256 # scale images to this size
CROP_SIZE=256 # then crop to this size
INPUT_NC=3  # number of channels in the input image
ASPECT_RATIO=1.0
 # change aspect ratio for the test images

# misc
GPU_ID=${1} # gpu id
NUM_TEST=25 # number of input images duirng test
NUM_SAMPLES=10 # number of samples per input images


# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./test.py \
  --dataroot ./datasets/${CLASS} \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir ${checkpoints_dir} \
  --direction ${DIRECTION} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --name ${NAME} \
  --input_nc ${INPUT_NC} \
  --num_test ${NUM_TEST} \
  --n_samples ${NUM_SAMPLES} \
  --aspect_ratio ${ASPECT_RATIO} \
  --epoch latest \
  --center_crop \
  --no_flip \
  --no_encode
