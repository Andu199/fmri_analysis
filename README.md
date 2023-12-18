# fmri_analysis
fMRI Volume Analysis using Deep Learning Methods

## Project structure:
* dataset/ - location of the data. This is an empty folder i which the database should be stored
* configs/ - folder containing config files (.yaml) for different tasks
* preprocess/ - folder containing scripts used to preprocess the data in order to be used by an ML model
  * fmriprep_wrapper.py - wrapper script over the fmriprep-docker command.
  * preprocess.py - a collection of functions which can be used to load, plot and compute connectomes and ROIs based on some specific functional atlases.