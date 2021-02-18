FROM python:3.6
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY models /models
COPY app.py /app.py
WORKDIR .

ENV PATH_TO_SERIALIZATION="/models"

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0"]