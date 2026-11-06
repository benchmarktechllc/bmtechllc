# Always-on Discord ask bot for the Hetzner VM.
# Build from projects/internal-tools/benchmark-cli-tool:
#   docker build -t benchmark-discord-ask .
#   docker run --env-file /etc/bmtech/discord-ask.env benchmark-discord-ask
FROM python:3.12-slim

WORKDIR /opt/benchmark-cli-tool

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e . \
    && mkdir -p artifacts \
    && useradd --create-home --uid 10001 bmtech \
    && chown -R bmtech:bmtech /opt/benchmark-cli-tool

USER bmtech

CMD ["benchmark-discord-ask"]
