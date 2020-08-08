docker run --gpus all -p 8501:8501 \
  --mount type=bind,source=/home/james/PycharmProjects/bert-tutorial/models/builds/0,target=/models/bert/0 \
  -e MODEL_NAME=bert -t tensorflow/serving
