# BASSMAN

BASSMAN (Best rAndom StarSpots Model calculAtioN) is the software to model the starspots on selected stars using photometry from the TESS satellite. The software is written in Python 3 by K. Bicz. For more details, see the paper on [Bicz et al. 2022](https://iopscience.iop.org/article/10.3847/1538-4357/ac7ab3). If you find this software useful in your work and you have some interesting results then we ask for the citation of the above article.

The software to work properly needs MacOS or Linux (for more advanced users WSL will also work). Recommended Python version between 3.6.8 - 3.11.*. To install the software you can use python codes in the direcory with data or put them in the folder where the enviromental wariable is added to the path. You have to put files claretld.dat and TESS_stars_params.dat to the /usr/local/bin/ folder. To install all of the needed libraries for Python run the install_python_modules.sh script. 

If you have python 3.9.5 or higher install starry ver. 1.2.0 instead of starry ver. 1.0.0 from the insttallation script.

File bassman_manual.pdf is a short instruction how to use each of the programs. Also programs bassman.py, tess_lightcurve.py, npz_plot.py and star_prep_bassman.py support -h and --help keys to display information about themselves. 

Short description of the programs inside package:
- tess_lightcurve.py is a simple program to download lightcurves from the TESS database,
- star_prep_bassman.py is a program that prepares the light curve downloaded by the tess_lightcurve for bassman,
- npz_plot.py is a simple program to plot seleted data from star_params.npz file created by star_prep_bassman,
- bassman.py is the final code to model the starspots on selected star.
