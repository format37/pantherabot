FROM python:3.13-slim

RUN useradd -m appuser

WORKDIR /server
COPY requirements.txt /server
RUN pip3 install -r requirements.txt --no-cache-dir
COPY panthera.py /server
COPY server.py /server
COPY tools_cli.py /server
RUN chown -R appuser:appuser /server

# Grant DAC_READ_SEARCH to Python so appuser can read root-owned photo dirs
RUN apt-get update -qq && apt-get install -y --no-install-recommends libcap2-bin \
    && setcap cap_dac_read_search+eip $(readlink -f $(which python3)) \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER appuser
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4221"]
