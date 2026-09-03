FROM python:3.13-slim

RUN useradd -m appuser

WORKDIR /server
COPY requirements.txt /server
RUN pip3 install -r requirements.txt --no-cache-dir
COPY panthera.py /server
COPY server.py /server
COPY tools_cli.py /server
COPY memory.py /server
RUN chown -R appuser:appuser /server

# No setcap here: appuser reads Telegram's root-owned photo dirs through an ACL
# on the host (see README), so the container needs no capability and can run
# with no-new-privileges.

USER appuser
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4221"]
