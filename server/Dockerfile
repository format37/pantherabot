FROM python:3.10

# Install Node.js (required by claude-agent-sdk / Claude CLI)
RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g @anthropic-ai/claude-code && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /server
COPY requirements.txt /server
RUN pip3 install -r requirements.txt --no-cache-dir
COPY panthera.py /server
COPY server.py /server
COPY tools_cli.py /server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4221"]
