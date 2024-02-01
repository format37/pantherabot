# get id of running container with image name
# container_id=$(sudo docker ps | grep pantherabot_server_1 | awk '{print $1}')
# echo "container id: $container_id"
# get logs
sudo docker logs -f panthera
# connect to container
# sudo docker exec -it $container_id /bin/bash