docker build . -t bert_builder-int-clf

# docker run -u "$(id -u):$(id -g)" --gpus all -it -exec --rm -v /home/james/PycharmProjects/build_history-a-bot:/home/james/build_history-a-bot build_history-a-bot-int-clf:latest /bin/bash

docker run --gpus all --rm -v /home/james/PycharmProjects/aBERT:/home/james/build bert_builder-int-clf:latest
