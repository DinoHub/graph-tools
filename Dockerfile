# FROM nvcr.io/nvidia/rapidsai/rapidsai:cuda10.2-base-ubuntu18.04
FROM nvcr.io/nvidia/rapidsai/rapidsai:21.10-cuda11.0-runtime-ubuntu18.04 
# FROM rapidsai/rapidsai:21.06-cuda11.0-runtime-ubuntu18.04-py3.7 

RUN mkdir -p /clustering
WORKDIR /clustering

COPY ./requirements.txt /clustering

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip
RUN pip install jupyter
RUN pip install --no-cache-dir -r requirements.txt

# RUN source activate rapids && \
#     pip install --upgrade pip && \
#     pip install jupyter && \
# #    pip install llvmlite --ignore-installed && \
#     pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
