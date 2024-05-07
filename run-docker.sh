# ------------------------------------------- run
docker-compose up

# ------------------------------------------- use however you like
docker ps --all
docker-compose exec <service_name> <command>
docker exec -it <container_id> /bin/bash

# ------------------------------------------- stop, clean up
docker-compose down

docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
docker rmi $(docker images -q)

yes | docker container prune
yes | docker image prune
yes | docker volume prune
yes | docker network prune
yes | docker system prune

# ------------------------------------------- check
docker ps --all
docker images
docker system df
docker volume ls
docker network ls
