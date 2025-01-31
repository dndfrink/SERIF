#!/bin/bash

redis-cli -h 127.0.0.1 -p 2223 shutdown
redis-cli -h 127.0.0.1 -p 2224 shutdown
redis-cli -h 127.0.0.1 -p 2225 shutdown
rm *.conf
rm dump.rdb