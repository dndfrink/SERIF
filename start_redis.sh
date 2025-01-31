#!/bin/bash

# Array of ports for the Redis instances
PORTS=(2223 2224 2225)

# Directory for Redis data (this will be different for each instance)
CONF_FILE_BASE="$PWD/redis"

# Loop to start each Redis server on a different port
for PORT in "${PORTS[@]}"; do 
  redis-server --bind 0.0.0.0 --port $PORT --cluster-enabled yes --cluster-config-file $CONF_FILE_BASE-$PORT.conf --cluster-node-timeout 5000 --protected-mode no &
  echo "Redis instance started on port $PORT with config $CONF_FILE"
done

# give redis servers time to come up
sleep 5

echo "All Redis instances are running."

redis-cli --cluster create 134.84.145.63:2223 134.84.145.63:2224 134.84.145.63:2225