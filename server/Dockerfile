FROM python:3.13-slim

RUN useradd -m appuser

WORKDIR /server
COPY requirements.txt /server
RUN pip3 install -r requirements.txt --no-cache-dir
COPY panthera.py /server
COPY server.py /server
COPY tools_cli.py /server
RUN chown -R appuser:appuser /server

USER appuser
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4221"]
