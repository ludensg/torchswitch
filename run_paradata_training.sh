#!/bin/bash

# Function to clean up processes
cleanup() {
    echo "Cleaning up processes..."
    ssh $MASTER_ADDR "pkill -f 'nc -lk $MASTER_PORT'"
    for ((i=1; i<=NODES; i++)); do
        NODE_NAME="ns0$i"
        [ "$i" -gt 9 ] && NODE_NAME="ns$i"
        ssh $NODE_NAME "pkill -f 'torchrun'"
    done
    echo "Cleanup done."
}

# Trap CTRL+C, CTRL+Z and quit signals
trap cleanup SIGINT SIGTSTP SIGQUIT

echo "Enter the number of nodes to use:"
read NODES

echo "Enter 0 to not use GPU nodes, 1 to use one GPU node, or 2 to use both GPU nodes:"
read GPU_CHOICE

# Define the master node and port
MASTER_ADDR="ns01"
MASTER_PORT="12345"
JOB_ID=$(date +%s)  # Unique job ID using the current timestamp

# Start listening on the master node in the background
echo "Starting listener on the master node ($MASTER_ADDR) on port $MASTER_PORT..."
ssh $MASTER_ADDR "nohup nc -lk $MASTER_PORT >/dev/null 2>&1 &"

# Give it a moment to start listening
sleep 2

# Activate the virtual environment and run the script on each node
for ((i=1; i<=NODES; i++)); do
    NODE_NAME="ns0$i"
    [ "$i" -gt 9 ] && NODE_NAME="ns$i"
    [ "$i" -eq 1 ] && [ "$GPU_CHOICE" != "0" ] && NODE_NAME="gpu1"
    [ "$i" -eq 2 ] && [ "$GPU_CHOICE" != "0" ] && NODE_NAME="gpu2"

    echo "Starting training on $NODE_NAME ..."
    # Use nohup and & to run the process in the background
    nohup ssh $NODE_NAME "source /home/gandelman/experiments/torchswitch/venv/bin/activate; echo 'Activated venv on $NODE_NAME'; \
    torchrun --nnodes=$NODES --nproc_per_node=1 --rdzv_id=$JOB_ID --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    paradata.py --nodes $NODES --gpu_choice $GPU_CHOICE" > training_node_$i.log 2>&1 &
    echo "Training launched on $NODE_NAME. Check training_node_$i.log for output."
done

echo "Training script executed on all nodes. Waiting for completion..."

wait
echo "Training completed on all nodes."
cleanup  # Perform cleanup