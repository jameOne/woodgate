FROM tensorflow/tensorflow:latest-gpu

COPY ./additional_requirements.txt /home/james/additional_requirements.txt

WORKDIR /home/james

RUN pip install --upgrade pip \
    && pip install -r additional_requirements.txt

RUN mkdir build

COPY . ./build

ENV MPLCONFIGDIR=/home/james/build/output

WORKDIR /home/james/build

CMD ["python", "main.py"]

#RUN python main.py
#
#FROM tensorflow/serving:latest-gpu
#
#COPY ./models/bert /models/bert
