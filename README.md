This README contains instructions for using `mpas_plot` utilities, including setting up the environment

# Environment setup

## Setting up conda
If you already have conda on your system, you can skip to the next section.

This utility includes a script that will set up a local install of conda to set up the needed
python environment. If you would rather use an existing conda install on your machine, skip to the
next section.

This script can only be used with bash or bash-like (e.g. ksh) shells. To use a different login
shell, you must configure conda manually.

```
source setup_conda.sh
```

## Build and load the conda environment

```
mamba env create -f environment.yml
conda activate mpas_plot
```

On subsequent logins after the first time you create the environment, you can simply run the last
command.

# Running the plotting script

The plotting script is built with argparse, so you can see a summary of the arguments by running with the --help (-h) flag:

```
$ python plot_mpas_netcdf.py -h
usage: plot_mpas_netcdf.py [-h] [-c CONFIG] [-d]
Script for plotting MPAS input and/or output in native NetCDF format
options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        File used to specify plotting options
  -d, --debug           Script will be run in debug mode with more verbose output
```

The config file is where you will specify all the various options for what you want to plot, including which files, variables, levels, etc you want to plot. To setup the script to use your specific options, you’ll need to create a configuration file (`config_plot.yaml`). An example file `config_plot.yaml.example` is provided for reference, and you can view all available options in the `default_options.yaml` file.

Once you have modified `config_plot.yaml` with all the settings you want, simply run the script:

```
$ python plot_mpas_netcdf.py
INFO:root:Reading data from /scratch2/BMC/fv3lam/MPAS_stoch/expt_dirs/test_stoch_global_plot_test/2023091500/forecast/history.2023-09-15_00.00.00.nc
INFO:root:Plotting variable t2m
INFO:root:Plotting first time step
INFO:root:Reading data from /scratch2/BMC/fv3lam/MPAS_stoch/expt_dirs/test_stoch_global_plot_test/2023091500/forecast/history.2023-09-15_01.00.00.nc
INFO:root:Plotting variable t2m
INFO:root:Plotting first time step
...
...
```

It may take some time to produce plots, depending on the size of your domain and number of fields plotted.

## Limitations

This plotting utility is in a very early form, and has several known limitations:

1. The user must know the name of the variable they want to plot, as well as the number of vertical levels if the variable has multiple.
2. Only the [PlateCarree](https://scitools.org.uk/cartopy/docs/latest/reference/projections.html#platecarree) projection is currently supported for output maps
3. The plotting script runs serially, which means it can take a long time to create a lot of large-domain plots.
4. Certain variables that have additional dimensions such as grid property values (e.g. kiteAreasOnVertex) may not work out-of-the-box.


