FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime

# Install dependencies
RUN pip install tb-nightly
RUN pip install future

RUN pip install scikit-image==0.16.2
RUN pip install tqdm==4.41.1
RUN pip install pyyaml==5.1.2
RUN pip install pytorch-pretrained-biggan==0.1.1
RUN pip install nltk==3.4.5