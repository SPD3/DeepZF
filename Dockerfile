FROM python:3.8.6
WORKDIR /DeepZF

RUN pip install --upgrade pip
# RUN pip3 install tensorflow==2.6.2
RUN pip3 install protein-bert==1.0.0
RUN pip3 install keras==2.6.0
RUN pip3 install protobuf==3.20.0
RUN pip3 install Keras-Preprocessing
RUN pip3 install ipython

RUN pip3 install -U scikit-learn scipy matplotlib
ENV LD_PRELOAD /usr/local/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

COPY BindZF_predictor/code/* .
CMD ["python3", "main_bindzfpredictor_predict.py", "-in", "40_zf_40_b.csv", "-out", "results.tsv", "-m", "model.p", "-e", "encoder.p", "-r", "1"]
