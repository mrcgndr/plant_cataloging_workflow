FROM continuumio/miniconda3:latest

WORKDIR /root

RUN mkdir plant_cataloging_workflow/

WORKDIR /root/plant_cataloging_workflow/

COPY . .

RUN mkdir data/ 
RUN mkdir data/ground_truth

RUN conda env create -f environment.yml
RUN echo "conda activate pcw" >> ~/.bashrc

ENTRYPOINT ["/bin/bash"]
