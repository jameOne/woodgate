FROM tensorflow/tensorflow:latest-gpu

RUN useradd -ms /bin/bash  james

COPY additional_requirements.txt \
    /home/james/additional_requirements.txt

WORKDIR /home/james

RUN pip install --upgrade pip \
    && pip install -r additional_requirements.txt \
    && mkdir build


ENV MPLCONFIGDIR=/home/james/build

WORKDIR /home/james/build

COPY . ./

CMD ["python", "main.py"]
