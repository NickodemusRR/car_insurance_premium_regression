FROM python:3.11.9-slim

WORKDIR /app
COPY ["requirements.txt", "./"] 

RUN pip install -r requirements.txt

COPY ["app.py", "predict_test.py", "model.bin", "./"] 

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "app:app"]