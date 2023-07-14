# BASSMAN

BASSMAN (Best rAndom StarSpots Model calculAtioN) is the software to model the starspots on selected stars using photometry from the TESS satellite. The software is written in Python 3 by K. Bicz. For more details, see the paper on [Bicz et al. 2022](https://iopscience.iop.org/article/10.3847/1538-4357/ac7ab3). If you find this software useful in your work and you have some interesting results then we ask for the citation of the above article.

The software to work properly needs MacOS or Linux (for more advanced users WSL will also work). Recommended Python version between 3.6.8 - 3.11.*. To install the software you can use python codes in the directory with data or put them in the folder where the environmental variable is added to the path. You have to put files claretld.dat and TESS_stars_params.dat to the /usr/local/bin/ folder. To install all of the needed libraries for Python run the install_python_modules.sh script. 

If you have python 3.9.5 or higher install starry ver. 1.2.0 instead of starry ver. 1.0.0 from the installation script. If you have Python 3.11 you should also install starry 1.2.0 and numpy version 1.23.1 (ignore the dependencies errors cause they do not affect the software). If you have arviz (only version <= 0.12.0 are supported) and you have some problem with the --sample key then just download the arviz.zip from the "possible_missing_files" folder and copy the missing files (they are usually connected with pymc3).

File bassman_manual.pdf is a short instruction how to use each of the programs. Also programs bassman.py, tess_lightcurve.py, npz_plot.py, and star_prep_bassman.py support -h and --help keys to display information about themselves. 

To check if the code is calculating everything right and is working you can check the star_params.npz file and example command in the folder "example".

Short description of the programs inside the package:
- tess_lightcurve.py is a simple program to download lightcurves from the TESS database,
- star_prep_bassman.py is a program that prepares the light curve downloaded by the tess_lightcurve for bassman,
- npz_plot.py is a simple program to plot selected data from star_params.npz file created by star_prep_bassman,
- bassman.py is the final code to model the starspots on selected star.
