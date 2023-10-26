FROM python:3.9.16
WORKDIR /server
# RUN bash -c 'apt-get install software-properties-common;add-apt-repository ppa:mc3man/trusty-media -y;apt-get update;apt-get install ffmpeg -y'
COPY requirements.txt /server
RUN pip3 install -r requirements.txt --no-cache-dir
# RUN python3 -m pip install --upgrade openai
COPY server.py /server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8443", "--ssl-keyfile=webhook_pkey.pem", "--ssl-certfile=webhook_cert.pem"]