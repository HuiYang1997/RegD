#!/bin/bash

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -model) model="$2"; shift ;;
        -dataset) dataset="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Debugging: Print parsed arguments
echo "Debug: Model is $model"
echo "Debug: Dataset is $dataset"

# Check if required arguments are set
if [ -z "$model" ]; then
    echo "Error: -model argument is required"
    exit 1
fi

if [ -z "$dataset" ]; then
    echo "Error: -dataset argument is required"
    exit 1
fi

lrs=(0.01 0.005 0.001 0.0005)
# Create a log file
log_file="results_others/grid_search_results_$model.log"
echo "Grid Search Results" > $log_file

# Define the parameter grid based on dataset
echo "Debug: Dataset is $dataset"
if [ "$dataset" == "mammal" ]; then
    dims=(2 5)
    batch_sizes=(32 64 128)
    epoch=400
else
    dims=(5 10)
    batch_sizes=(1024 2048)
    if [ "$dataset" == "noun" ]; then
      epoch=400
    else
      epoch=500
    fi
fi


# Iterate over all combinations of parameters
for dim in "${dims[@]}"; do
  for lr in "${lrs[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
      # Debugging: Print the current dataset and parameters
      echo "Dataset: $dataset"
      echo "Dims: $dim"
      echo "Batch sizes: $batch_size"
      echo "Learning rate: $lr"

      # Construct the command string
      command="python train_others.py -debug 1 -dataset $dataset \
                 -train_non_basic_percent 0 -model $model \
                 -dim $dim -epoch $epoch \
                 -lr $lr -eval_method energy -loss_type vanilla \
                 -batch_size $batch_size -grid_search 1"

      # Debugging: Print the command to be executed
      echo "Running: $command"

      # Execute the command
      eval $command

      # Log the results
      echo "Command: $command" >> $log_file
      echo "----------------------------------------" >> $log_file
    done
  done
done