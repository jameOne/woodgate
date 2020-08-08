docker build . -t bert_builder-int-clf

# docker run -u "$(id -u):$(id -g)" --gpus all -it -exec --rm -v /home/james/PycharmProjects/build-a-bot:/home/james/build-a-bot build-a-bot-int-clf:latest /bin/bash

docker run --gpus all --rm -v /home/james/PycharmProjects/aBERT:/home/james/build bert_builder-int-clf:latest
