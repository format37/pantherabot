FROM python:3.9.16
WORKDIR /server
COPY requirements.txt /server
RUN pip3 install -r requirements.txt --no-cache-dir
# RUN python3 -m pip install pip install -U langchain-community tavily-python
# RUN python3 -m pip install pip install -U tavily-python
RUN python3 -m pip install tavily-python==0.3.3
RUN python3 -m pip install langchain-experimental==0.0.59
COPY panthera.py /server
COPY server.py /server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4221"]