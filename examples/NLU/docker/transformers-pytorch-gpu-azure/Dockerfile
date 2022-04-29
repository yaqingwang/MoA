FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04
LABEL maintainer="Hugging Face"
LABEL repository="transformers"



RUN mkdir /app
WORKDIR /app

RUN ldconfig /usr/local/cuda/lib64/stubs && \
    pip install --no-cache-dir GPUtil && \
    pip install --no-cache-dir azureml-defaults

RUN pip install --no-cache-dir \
    mkl \
    torch


#
#RUN git clone https://github.com/NVIDIA/apex
#RUN cd apex && \
#    python setup.py install && \
#    python -m pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

COPY . transformers/
RUN cd transformers/ && \
    python -m pip install --no-cache-dir .
## NCCL 2.4 does not work with PyTorch, uninstall
#RUN apt-get update && apt-get --purge remove libnccl2 -y --allow-change-held-packages

#RUN apt-get -y update && apt-get -y install --no-install-recommends libnccl2=2.3.7-1+cuda10.0 libnccl-dev=2.3.7-1+cuda10.0

# Create a working directory
#RUN mkdir /app
#WORKDIR /app

# Create a non-root user and switch to it
#RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
# && chown -R user:user /app
##RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
#USER user

# All users can use /home/user as their home directory
#ENV HOME=/home/user
#RUN chmod 777 /home/user
#
#ENV CONDA_AUTO_UPDATE_CONDA=false
#ENV PATH=/home/user/miniconda/bin:$PATH
#RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh \
# && chmod +x ~/miniconda.sh \
# && ~/miniconda.sh -b -p ~/miniconda \
# && rm ~/miniconda.sh \
# && conda install python==3.6.13 \
# && conda install ruamel.yaml \
# && conda clean -ya




ADD /requirements.txt /app/

RUN pip install -r requirements.txt

RUN conda install pip=20.1.1
RUN pip install --upgrade azureml-sdk[notebooks,contrib]
RUN pip install cryptography==3.4.1




CMD ["/bin/bash"]

