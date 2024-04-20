#!/bin/bash
# Copyright 2020 Snowdar
#           2022 Jianchen Li

# debug mode
pdb=false

. parse_options.sh

if [[ $# < 1 ]]; then
    echo "[exit] Num of parameters is zero, expected a launcher."
    echo "usage: $0 <launcher> [launcher-options]"
    echo "e.g. $0 speakernet/pipelines/train.py --gpu-id=0,1,2"
    exit 1
fi

launcher=$1
shift

[ ! -f $launcher ] && echo "Expected $launcher (*.py) to exist." && exit 1

# Should note the " and space char when giving a parameter from shell to python.
launcher_options=""
num_gpu=1
while true; do
    [ $# -eq 0 ] && break

    if [[ $1 == "--gpu-id="* ]]; then
        gpu_id_option=$(echo "$1" | sed 's/ /,/g')
        launcher_options="$launcher_options $gpu_id_option"
        num_gpu=$(echo $gpu_id_option | awk -F '=' '{print $2}' | sed 's/[,-]/\n/g' | sed '/^$/d' | wc -l)
    else
        launcher_options="$launcher_options $1"
    fi
    shift
done

# Add multi-gpu case.
if [ $num_gpu -gt 1 ]; then
    train_cmd="torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpu"
else
  if [[ "$pdb" == 'true' ]]; then
    train_cmd="ipdb3"
  else
    train_cmd="python3"
  fi
fi

$train_cmd $launcher $launcher_options || exit 1 

exit 0
