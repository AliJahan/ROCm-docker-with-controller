#!/bin/bash

docker compose build $1  --no-cache  --progress plain &> $1_build.log 
