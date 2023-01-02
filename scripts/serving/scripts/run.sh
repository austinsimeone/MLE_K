#!/bin/sh

. scripts/serving/scripts/console.sh

info "💣 Killing Server Docker containers"
docker-compose kill server 2>/dev/null
info "🔥 Removing Server Docker containers"
docker-compose rm -f server 2>/dev/null
info "⚡️ Starting Server Docker container"
docker-compose -f scripts/serving/docker-compose.yml up --force-recreate