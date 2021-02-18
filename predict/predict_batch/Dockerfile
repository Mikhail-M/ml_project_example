FROM python:3.6
RUN pip install scikit-learn==0.24.1 pandas numpy click
COPY models /models
COPY batch_predict.py /batch_predict.py

WORKDIR .

ENV PATH_TO_SERIALIZATION="/models"
ENV OUTPUT="output/output.csv"
ENV PATH_TO_DATA="input/test.csv"

ENTRYPOINT ["python", "batch_predict.py"]

