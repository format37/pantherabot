source .env
sudo -E docker compose down -v
sudo -E docker compose up --build --force-recreate -d