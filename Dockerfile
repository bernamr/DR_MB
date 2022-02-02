# write some code to build your image

FROM python:3.8.6-buster

COPY api /api
COPY DR_MB /DR_MB
COPY model.tflit /model.tflite
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
