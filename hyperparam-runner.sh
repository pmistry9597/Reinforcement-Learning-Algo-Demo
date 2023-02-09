#!/bin/bash
index=-1
code=0

# Just meant to run hyperparam-search.py, except in separate iterations so python can be cleaned up every time.

while [ $code == 0 ]
do
    index=$(($index+1))
    echo Running hyperparam index $index
    python -u hyperparam-search.py $index
    code=$?
    retry=0
    while [ $code != 0 ] && [ $retry -lt 5 ]
    do
        python -u hyperparam-search.py $index
        code=$?
        retry=$(($retry+1))
    done
done

echo Search concluded: $index $code