#!/bin/bash

for dir in */; do
    if [[ ! $dir == _* ]]; then # Ignore dirs begining with underscore
        dir=${dir%/}
        cd "$dir" || { echo "Failed to cd into $dir"; exit 1; }
        ./run.sh
        cd ..
    fi
done
