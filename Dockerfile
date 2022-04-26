FROM continuumio/miniconda3:latest

WORKDIR /root

RUN mkdir pcw/

WORKDIR /root/pcw/

COPY . .

RUN mkdir data/ 
RUN mkdir data/ground_truth

RUN conda env create -f environment.yml
RUN echo "conda activate pcw" >> ~/.bashrc

ENTRYPOINT ["/bin/bash"]
