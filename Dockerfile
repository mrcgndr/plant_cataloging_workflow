FROM python:3.9.12

WORKDIR /root

COPY . .

RUN mkdir data/ 
RUN mkdir data/ground_truth

RUN pip install -U pip

RUN pip install .

ENTRYPOINT [ "/bin/bash" ]
