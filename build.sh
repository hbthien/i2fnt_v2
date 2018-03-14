#download data and convert data to TFRecord format (tokenize strings, generate word_counts

MSCOCO_DIR="./data"
./data/download_and_preprocess_mscoco.sh $MSCOCO_DIR
python ./data/build_mscoco_data.py



#Download the Inception v3 Checkpoint

INCEPTION_DIR="./data/inception"
mkdir -p ${INCEPTION_DIR}

wget "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
tar -xvf "inception_v3_2016_08_28.tar.gz" -C ${INCEPTION_DIR}
rm "inception_v3_2016_08_28.tar.gz"


#Train a Model

INCEPTION_CHECKPOINT="${INCEPTION_DIR}/inception_v3.ckpt"
MODEL_DIR="./model"
mkdir -p $MODEL_DIR

python train.py \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=1000000
