FROM tensorflow/tensorflow:0.12.1-devel-gpu-py3
MAINTAINER Shreyas Hukkeri <hshreyas12@gmail.com>


RUN pip3 install numpy scipy matplotlib pydicom scikit-learn pypng h5py
RUN pip3 install git+https://github.com/tflearn/tflearn.git

WORKDIR /
COPY train.sh .
COPY train.py .
COPY dbfile.py .
