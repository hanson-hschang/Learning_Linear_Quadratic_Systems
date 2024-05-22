# Learning Linear Quadratic Systems

## Introduction

This project includes code for running linear quadratic games and linear quadratic Gaussian simulations. It also includes scripts for plotting the results of these simulations.

## Requirements

This project requires Python 3.x and the following Python packages:

- numpy
- matplotlib
- scipy
- pandas
- pytorch
- numba
- seaborn
- control

You can install these packages using pip or conda.


## Usage

### Linear Quadratic Gusasian
To run the linear quadratic guassian, navigate to the following directory and run the python script.
```
cd linear_quadratic_gussian/adaptive
python neurips.py
```

Note that the original main file is `regret_comparison.py`. 
One may run this file to obtain result from [original paper: Krauth *et. al.*, 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/aaebdb8bb6b0e73f6c3c54a0ab0c6415-Abstract.html).

The `neurips.py` file runs the simulation, save simulation information in the .log file and stored simulation data.

To plot the result for our preliminary neurips results, run the following python script.
```
python neurips_data_plot.py
```

### Linear Quadratic Game
To run the linear quadratic game, navigate to the following directory and run the python script.
```
cd linear_quadratic_games
python sim_rum.py
```

It runs the simulation, save simulation information in the .log file and stored simulation data.

To plot the result for the [original paper: Zhang *et. al.*, 2021](https://openreview.net/forum?id=NVAOPWZWYlv), run the following python script.
```
python sim_plot.py
```
or run the other plotting file to have our preliminary neurips results.
```
python sim_plot_result.py
```

### Compared Plots with dual-EnKF
One may found the results of dual-EnKF results from [here](https://drive.google.com/drive/folders/1tWFHcO6EF1lOcfO2MgSbyHYb9_Iy6xfT?usp=sharing). Please download the folder and place it directly under the main directory. Once its done, use the following command to plot the result.
```
python neurips_data_plot.py
```


Some other simulated results may also found [here](https://drive.google.com/drive/folders/1h6lkcsROXrtMZp1lIEr1OHIs7166QX1H?usp=sharing).
 