#!/bin/bash
# Deploy: rebuild and (re)create the bot container. Run as the normal user: ./compose.sh
cd "$(dirname "$0")" || exit 1

. ./.env

# Resolve the real user's home even under sudo: plain `sudo` resets HOME=/root,
# so `~` in the .claude bind mounts would silently point at stale/empty /root
# state and the Claude CLI dies at startup ("Control request timeout:
# initialize" outage, 2026-07-10). docker-compose.yml requires CLAUDE_HOME.
REAL_HOME=$(getent passwd "${SUDO_USER:-$(id -un)}" | cut -d: -f6)
export CLAUDE_HOME="${CLAUDE_HOME:-$REAL_HOME}"

sudo -E docker compose down -v
sudo -E docker compose up --build --force-recreate -d
