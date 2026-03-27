TAG="02"
docker build --no-cache -f docker/Dockerfile.lrzcluster -t lerobot-lrzcluster .
docker tag lerobot-lrzcluster olivertum/lerobot-lrzcluster:${TAG}
docker push olivertum/lerobot-lrzcluster:${TAG}