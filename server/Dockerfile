FROM python:3.10
WORKDIR /server
COPY requirements.txt /server
RUN pip3 install -r requirements.txt --no-cache-dir
COPY panthera.py /server
COPY server.py /server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4221"]