#!/bin/bash

# Parameter arrays
connections=(20 10 5)
particle_counts=(500 100 50 10)

# Loop through all combinations
for conn in "${connections[@]}"; do
    for particles in "${particle_counts[@]}"; do
        echo "Running with connections=${conn}, particles=${particles}"
        python mppi.py --number-of-connections "${conn}" --number-of-samples "${particles}"
        mv mppi "mppi_masses_${conn}_particles_${particles}"
    done
done