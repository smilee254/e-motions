#!/bin/bash
echo "Downloading DB-IP City Lite database..."
curl -L https://cdn.jsdelivr.net/npm/dbip-city-lite/dbip-city-lite.mmdb.gz -o api/dbip-city-lite.mmdb.gz
echo "Extracting..."
gunzip -f api/dbip-city-lite.mmdb.gz
echo "Done! Database ready at api/dbip-city-lite.mmdb"
