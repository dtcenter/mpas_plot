This README contains instructions for using `mpas_plot` utilities, including setting up the environment

For detailed examples of different plotting functions, see the GitHub wiki:
https://github.com/dtcenter/mpas_plot/wiki/mpas_plot-config-examples-and-use-cases

# Environment setup

## Setting up conda and mpas_plot conda environment
This utility includes a script that will set up a local install of conda to set up the needed
python environment. If you would rather use an existing conda install on your machine, an ``environment.yml`` recipe
file is provided for you to create your own environment, skipping these steps.

This script can only be used with bash or bash-like (e.g. ksh) shells. To use a different login
shell, you must configure conda manually.

```
source setup_conda.sh
conda activate mpas_plot
```

# Running the plotting script

The plotting script is built with argparse, so you can see a summary of the arguments by running with the --help (-h) flag:

```
$ python plot_mpas_netcdf.py -h
usage: plot_mpas_netcdf.py [-h] [-c CONFIG] [-d] [-p PROCS]

Script for plotting a custom field on the native MPAS grid from native NetCDF format files

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        File used to specify plotting options
  -d, --debug           Script will be run in debug mode with more verbose output
  -p PROCS, --procs PROCS
                        Number of processors for generating plots in parallel
```

The config file is where you will specify all the various options for what you want to plot, including which files, variables, levels, etc you want to plot. To setup the script to use your specific options, you’ll need to create a configuration file (`config_plot.yaml`). An example file `config_plot.yaml.example` is provided for reference, and you can view all available options in the `default_options.yaml` file.

Once you have modified `config_plot.yaml` with all the settings you want, simply run the script:

```
$ python plot_mpas_netcdf.py
INFO     Loading user config settings
INFO     Loading data from netcdf files
INFO     Setting up plot tasks
INFO     Submitting to starmap
INFO     Starting plotit() for var='t2m', lev=0
INFO     PlateCarree does not use standard_parallels; ignoring
INFO     One or more latitude/longitude range values were not set; plotting full projection
INFO     Done saving plot tutorial_data/t2m_lev0_2025-05-08_00:00:00.png. Plot generation 10.880536556243896 seconds
INFO     Starting plotit() for var='precipw', lev=0
...
...
```

It may take some time to produce plots, depending on the size of your domain and number of fields plotted.

## Custom colormaps
Some custom colormaps have been set up to mimic the typical way of displaying certain data. These colormaps are specified in YAML format files in the `colormaps/` subdirectory.
These custom colormaps may overwrite some user settings from `config_plot.yaml`, so if you want different settings you will have to modify the files there.

For example, `colormaps/radar_refl.yaml` is designed to mimic the typical display scheme for radar reflectivity data, so the following settings are included:

```
vmin: 5
vmax: 75
plot_under: False
```

This limits the color range to data between 5 and 75 (dBz), and `plot_under: False` specifies that data below vmin (5) should not be plotted at all; data above vmax (75) will be
plotted if present but will have the same color value as 75.

## Limitations

This plotting utility has no funding for technical support, and has several known limitations:

1. The user must know the name of the variable(s) they want to plot.
2. Interpolating to projections other than [PlateCarree](https://scitools.org.uk/cartopy/docs/latest/reference/projections.html#platecarree) (lat-lon projection) is very resource-intensive and slow.
3. Certain variables that have additional dimensions such as grid property values (e.g. kiteAreasOnVertex) may not work out-of-the-box.


