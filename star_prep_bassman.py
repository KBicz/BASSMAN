#! /usr/bin/python3

import os
import matplotlib
import numpy as np
import matplotlib.pylab as plt

from sys import argv
from scipy import stats
from math import asin, degrees, pi
from scipy.ndimage import median_filter
from scipy.interpolate import CubicSpline
from matplotlib.ticker import ScalarFormatter
from matplotlib.widgets import RectangleSelector

def helpf():
    print("\n  Program star_prep_bassman.py for MacOS and Linux written by K. Bicz, ver. of Nov 23, 2021.")
    print("  Preparing parameters of the star for modelling stellar spots.\n")
    print("  Usage: star_prep_bassman.py <-tic=str> [-out=file] [-lc=file] [-edit=file] [-model=file] [-ylm=int]")
    print("         [-dinc=float] [-p0=float] [-rmflares[=float]] [-niter=int] [-npoints=int] [-sigma=float]")
    print("         [--fulltime] [-rmparts] [--noerr] [--disp] [--force]")
    print()
    print("         option -tic : tic number of star.")
    print("                -out : destination file for parameters of star (default out = star_params.npz).")
    print("                -lc : light curve of star (default lc = lc.data).")
    print("                -edit : edit previously selected part of light curve (default edit = star_params.npz).")
    print("                -model : plot given model on selected part of light curve (default model = model.dat).")
    print("                -sigma : sigma for zscore outliers (default sigma = 1.9).")
    print("                -ylm : spherical harmonic degree (default ylm = 20).")
    print("                -dinc : default inclination if vsini not availeable (default dinc = 50).")
    print("                -p0 : plot as dashed lines moments of time corresponing to given moment of phase = 0.")
    print("                -rmflares : uses median filter to remove outliers (sigma = 0.05).")
    print("                -npoints : works only with rmflares, n of points for median filer.")
    print("                -niter : works only with rmflares, number of repetitive usages of the median filter.")
    print("                --fulltime : Take whole light curve, not selected time interval.")
    print("                --rmparts : select and remove any number of parts of the light curve.")
    print("                --noerr : Forces not to remove outliers.")
    print("                --disp : Display selected interval after z score outliers remove.")
    print("                --force : Forces to use default inclinaction (dinc value).")
    print()
    exit()

def str_to_num(string,ntype,name):
    try:
        return ntype(string)
    except ValueError:
        print("\a# Error! {} has to be number!".format(name))
        exit()

def rmflares(time,flux,sigmarmf,displayctrl,nctrl,npts,mintim,niter,fluxo):
    
    if nctrl: nel = npts
    else: nel = int(len(flux)**0.5*1.1)
    if nel%2==0: nel -= 1

    print("\n> Using {:d} points in median filter.".format(nel))
    
    flux2 = median_filter(flux,nel,mode='mirror')
    for i in range(len(flux2)):
        if flux2[i] == 0:
            flux2[i] = flux[i]
    dataprod = flux/flux2
    dane = flux[np.abs(stats.zscore(dataprod)) <= sigmarmf]
    czas = time[np.abs(stats.zscore(dataprod)) <= sigmarmf]
    flux2 = flux2[np.abs(stats.zscore(dataprod)) <= sigmarmf]
    dataprod = dataprod[np.abs(stats.zscore(dataprod)) <= sigmarmf]
    flux2 *= dataprod

    cs = CubicSpline(czas,flux2,bc_type='natural')
    flux3 = cs(time)

    flux4 = []
    for i in range(len(time)):
        if time[i] in czas:
            flux4.append(flux[i])
        elif time[i] not in czas:
            flux4.append(flux3[i])
    flux4 = np.array(flux4)

    if displayctrl:
        plt.figure(figsize=(12,7))
        plt.plot(time+mintim, fluxo,'C0.',ms=1,label="Data")
        plt.plot(czas+mintim, dane,'C1.',ms=1)
        plt.plot(time+mintim, flux4,"C3.",ms=1,label = "Data after filtration (npoints = {}, iterations = {})".format(nel,niter))
        plt.gca().get_xaxis().set_major_formatter(ScalarFormatter(useOffset=False))
        plt.gca().get_yaxis().set_major_formatter(ScalarFormatter(useOffset=False))
        plt.xlabel("TBJD [days]")
        plt.ylabel("Normalized flux")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.075), ncol=2,numpoints=5)
        plt.show()

    return flux4

def line_select_callback(eclick, erelease):
    global x1, y1, x2, y2
    x1 = -666
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

def toggle_selector(event):
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        toggle_selector.RS.set_active(True)

def select_area(current_ax,green = False):
    if green: toggle_selector.RS = RectangleSelector(current_ax, line_select_callback, drawtype='box', useblit=True,button=[1, 3],minspanx=5,minspany=5,spancoords='pixels',interactive=True,rectprops = dict(facecolor='C2', edgecolor = 'black', alpha=0.2, fill=True))
    else: toggle_selector.RS = RectangleSelector(current_ax, line_select_callback, drawtype='box', useblit=True,button=[1, 3],minspanx=5,minspany=5,spancoords='pixels',interactive=True)
    plt.connect('key_press_event',toggle_selector)

def prepare(tic,file,lc,sigma,ydeg,udeg,amp,dinc,full,display,force,noerr,nctrl,npts,sigmarmf,rmfctrl,niter,rmpartsctrl,p0,p0ctrl,modelfile,modelctrl,editfile,editctrl):
    try:
        stars = np.genfromtxt("/usr/local/bin/TESS_stars_params.dat",dtype='U13,U6,U10,U10,U10,U10,U10',names=['tic','vsini','rad','pls','prot','temp','logg'])
    except:
        print("\a# Error! No file TESS_stars_params.dat in folder /usr/local/bin")
        exit()

    wh = np.where(stars['tic'] == tic)
    vsini = stars['vsini'][wh][0]
    rad = stars['rad'][wh][0]
    pls = stars['pls'][wh][0]
    prot = stars['prot'][wh][0]
    temp = stars['temp'][wh][0]
    logg = stars['logg'][wh][0].lower()
    if prot == 'nan': prot = pls

    if not editctrl:
        data = np.genfromtxt(lc,dtype="float,float,float,float",names=["time","flux","err","sth"])
        flux = data['flux']/1000.0+1.0
        err = data['err']/1000.0
        time = data['time']-min(data['time'])
        orf = flux.copy()
        ore = err.copy()
        ort = time.copy()
    else:
        data = np.load(editfile, allow_pickle=True)
        flux = data['flux']
        err = data['sigma']
        time = data['t']
        orf = data['fluxorig']/1000+1
        ore = data['errorig']
        ort = data['timeorig']
        ort -= ort[0]
        mintim2 = data['mintim']+time[0]
        time -= time[0]

    if modelctrl:
        model = np.genfromtxt(modelfile,dtype='float,float,float,float',names=['tbjd','flux','err','model'])
        model['model'] = model['model']/1000.0+1.0
        if not editctrl: model['tbjd'] -= model['tbjd'][0]

    if rmfctrl and not editctrl:
        for _ in range(niter-1): flux = rmflares(time,flux,sigmarmf,False,nctrl,npts,min(data['time']),niter,data['flux'])
        flux = rmflares(time,flux,sigmarmf,display,nctrl,npts,min(data['time']),niter,data['flux']/1000+1)

    if not full and not editctrl:
        fig, current_ax = plt.subplots(figsize=(12,7))
        fig.canvas.set_window_title('Select area')
        plt.get_current_fig_manager().window.wm_geometry("+0+190")
        plt.subplots_adjust(left = .14, top = .88, right = .95)
        plt.gca().get_xaxis().set_major_formatter(ScalarFormatter(useOffset=False))
        plt.gca().get_yaxis().set_major_formatter(ScalarFormatter(useOffset=False))
        current_ax.plot(time+min(data['time']),flux,'C0.',ms=1,label='Data')
        current_ax.set_xlabel("TBJD [days]")
        current_ax.set_ylabel("Normalized flux")
        current_ax.set_xlim(min(data['time'])-1,max(data['time'])+1)
        current_ax.set_ylim(min(flux),max(flux))
        if p0ctrl:
            isline = False
            tstart = p0
            if p0 == min(data['time']): pass
            elif p0 > min(data['time']):
                while tstart > min(data['time']): tstart -= float(prot)
                tstart += float(prot)
            elif p0 < min(data['time']):
                while tstart < min(data['time']): tstart += float(prot)
            i = 1
            while tstart <= max(data['time']):
                if not isline:
                    plt.plot([tstart,tstart],[min(flux),max(flux)],'k--',label="Moment matching p0 = {} TBJD".format(p0))
                    isline = True
                else:
                    plt.plot([tstart,tstart],[min(flux),max(flux)],'k--')
                plt.text(tstart+0.4*float(prot),max(flux)/2*1.99,str(i),fontsize=15)
                tstart += float(prot)
                i += 1
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.075), ncol=2,numpoints=5)

        select_area(current_ax)
        plt.show()

        try:
            x1s = x1-min(data['time'])
            x2s = x2-min(data['time'])
            timesav = time[(time >= x1s) & (time <= x2s)]
            fluxsav = flux[(time >= x1s) & (time <= x2s)]
            errsav  = err[(time >= x1s) & (time <= x2s)]
            if modelctrl:
                modelf = model['model'][(time >= x1s) & (time <= x2s)]
                modelt = model['tbjd'][(time >= x1s) & (time <= x2s)]
        except NameError:
            print("\a# Error! There is no selected time interval!\n")
            exit()
    elif full and not editctrl:
        x1s = 0
        x2s = len(data['time']) 
        timesav = time
        fluxsav = flux
        errsav = err
        if modelctrl:
            modelf = model['model'][(time >= x1s) & (time <= x2s)]
            modelt = model['tbjd'][(time >= x1s) & (time <= x2s)]
    elif editctrl:
        timesav = time
        fluxsav = flux
        errsav = err
        if modelctrl:
            modelf = model['model'][(model['tbjd'] >= data['timeorig'][0]) & (model['tbjd'] <= data['timeorig'][-1])]
            modelt = model['tbjd'][(model['tbjd'] >= data['timeorig'][0]) & (model['tbjd'] <= data['timeorig'][-1])]
            modelt -= min(data['timeorig'])
            model['tbjd'] -= min(data['timeorig'])

    if rmpartsctrl:
        nparts = 0
        if not editctrl: fluxsavo, timesavo = fluxsav.copy(), timesav.copy()
        else: fluxsavo, timesavo = data['fluxorig'].copy()/1000.0+1.0, data['timeorig'].copy()-min(data['timeorig'])
        x1o,x1p = 0, 1
        x2o, x2p = 0, 1
        while x1o != x1p and x2o != x2p:
            try:
                x1o = x1p
                fig, current_ax = plt.subplots(figsize=(12,7))
                fig.canvas.set_window_title('Select area no. {} to remove.'.format(nparts+1))
                plt.get_current_fig_manager().window.wm_geometry("+0+190")
                plt.subplots_adjust(left = .14, top = .88, right = .95)
                plt.gca().get_xaxis().set_major_formatter(ScalarFormatter(useOffset=False))
                plt.gca().get_yaxis().set_major_formatter(ScalarFormatter(useOffset=False))
                if not editctrl:
                    current_ax.plot(timesavo+min(data['time']),fluxsavo,'C3.',ms=2,label="Removed data")
                    current_ax.plot(timesav+min(data['time']),fluxsav,'C0.',ms=2,label="Selected data")
                    if modelctrl: current_ax.plot(modelt+min(data['time']),modelf,'C1.',ms=2,label='Given flux model')
                else:
                    current_ax.plot(timesavo+mintim2 ,fluxsavo,'C3.',ms=2,label="Removed data")
                    current_ax.plot(timesav+mintim2,fluxsav,'C0.',ms=2,label="Selected data")
                    if modelctrl: current_ax.plot(modelt+mintim2 ,modelf,'C1.',ms=2,label='Given flux model')
                current_ax.set_xlabel("TBJD [days]")
                current_ax.set_ylabel("Normalized flux")
                if not editctrl: current_ax.set_xlim(min(timesavo+min(data['time']))-1,max(timesavo+min(data['time']))+1)
                else: current_ax.set_xlim(min(timesavo+mintim2)-1,max(timesavo+mintim2)+1)
                current_ax.set_ylim(min(fluxsavo),max(fluxsavo))
                select_area(current_ax,True)
                plt.legend(numpoints=5)
                plt.show()

                if not editctrl:
                    x1p = x1-min(data['time'])
                    x2p = x2-min(data['time'])
                else:
                    x1p = x1-mintim2
                    x2p = x2-mintim2
                if x1o != x1p and x2o != x2p:
                    if (len(np.where((timesav > x1p) & (timesav < x2p))[0]) == 0 or len(np.where((timesav > x1p) & (timesav < x2p))[0]) < len(np.where((ort > x1p) & (ort < x2p))[0])): 
                        wh = np.where((ort >= x1p) & (ort <= x2p))
                        timesav = np.array([*timesav, *ort[wh]])
                        sorti = np.argsort(timesav)
                        timesav = timesav[sorti]
                        fluxsav = np.array([*fluxsav, *orf[wh]])
                        errsav = np.array([*errsav, *ore[wh]])
                        fluxsav = fluxsav[sorti]
                        errsav = errsav[sorti]
                    else:
                        fluxsav = np.array([*fluxsav[np.where(timesav < x1p)],*fluxsav[np.where(timesav > x2p)]])
                        errsav  = np.array([*errsav[np.where(timesav < x1p)],*errsav[np.where(timesav > x2p)]])
                        timesav = np.array([*timesav[np.where(timesav < x1p)],*timesav[np.where(timesav > x2p)]])

                nparts += 1
            except:
                print("# Error! Try to select observations within observed time period!")

    if not noerr and not editctrl:
        timesav = timesav[np.abs(stats.zscore(fluxsav)) < sigma]
        errsav  = errsav[np.abs(stats.zscore(fluxsav)) < sigma]
        fluxsav = fluxsav[np.abs(stats.zscore(fluxsav)) < sigma]

    if display:
        plt.figure(figsize=(12,7))
        if not editctrl:
            plt.plot(time+min(data['time']),flux,"C0.",ms=1,label='Complete light curve')
            plt.plot(timesav+min(data['time']),fluxsav,'C1.',ms=1,label='Selected interval')
        else:
            plt.plot(time+mintim2,flux,"C0.",ms=1,label='Complete light curve')
            plt.plot(timesav+mintim2,fluxsav,'C1.',ms=1,label='Selected interval')
        plt.xlabel("TBJD [days]")
        plt.ylabel("Normalized flux")
        plt.gca().get_xaxis().set_major_formatter(ScalarFormatter(useOffset=False))
        plt.gca().get_yaxis().set_major_formatter(ScalarFormatter(useOffset=False))
        plt.legend(fontsize=10, numpoints=5)
        plt.show()

    if temp!='nan': temp = float(temp)
    if logg!='nan': logg = float(logg)

    try:
        if str(temp) != "nan" and str(logg) != 'nan':
            try:
                tempers = np.genfromtxt("/usr/local/bin/claretld.dat",dtype='float,float,float,float',names=['logg','temp','u1','u2'])
            except:
                print("\a# Error! No file claretld.dat in folder /usr/local/bin")
                exit()
            tempcl = int(round(temp / 100.0)) * 100
            loggcl = round(logg * 2) / 2
            whlogg = np.where(tempers['logg']==loggcl)
            temps = tempers['temp'][whlogg]
            u1t = tempers['u1'][whlogg]
            u2t = tempers['u2'][whlogg]
            wht = np.abs(temps-tempcl).argmin()
            u1, u2 = u1t[wht], u2t[wht]
        else: u1, u2 = 0.40, 0.25
    except:
        u1, u2 = 0.40, 0.25

    B = dict(
    ydeg=ydeg,  # degree of the map
    udeg=udeg,  # degree of the limb darkening
    inc=dinc,  # inclination in degrees
    amp=amp,  # amplitude (a value prop. to luminosity)
    r=float(rad),  #  radius in R_sun
    prot=float(prot),  # rotational period in days
    u=[u1, u2],  # limb darkening coefficients
    temp=temp,
    )

    if prot!='nan' and pls!='nan':
        if abs(float(pls)-float(prot)) > 0.5:
            print("\n\a% Warning! P_rot = {} and P_rot_ls = {} - variability may not be caused by spots.".format(prot,pls))

    if force != True:
        if vsini != 'nan' and prot != 'nan' and rad != 'nan':
            try:
                inc = degrees(asin(float(vsini)*float(prot)*24.0*60.0*60.0/(2*pi*float(rad)*696340.0)))
                print("\n> Estimated inclination = {:.2f} degrees.\n".format(inc))
            except ValueError:
                print("\a\n# Error! Cannot calculate arcsin({:.2f})!".format(float(vsini)*float(prot)*24.0*60.0*60.0/(2*pi*float(rad)*696340)))
                print("\n\a% Warning! Program can't estimate inclination.")
                B['inc'] = dinc
                print("> Inclination takes default value = {:.2f} degrees.\n".format(B['inc']))
        elif vsini == 'nan' or prot != 'nan' or rad != 'nan':
            print("\a\n# Error! Missing values!")
            print("  vsini = {} []".format(vsini))
            print("  p_rot = {} [days]".format(prot))
            print("  radius = {} [R_sun]".format(rad))
            print("  temperature = {} [K]".format(temp))
            if p0ctrl: print("  p0 = {} [days]\n".format(p0))
            B['inc'] = dinc
            print("> Inclination takes default value = {:.2f} degrees.\n".format(B['inc']))

    if not editctrl: 
        np.savez(file, B=B, t=timesav, flux=fluxsav, sigma=errsav, fluxorig = data['flux'][(time >= x1s) & (time <= x2s)], errorig = data['err'][(time >= x1s) & (time <= x2s)], timeorig = data['time'][(time >= x1s) & (time <= x2s)], mintim = min(data['time']),ort=ort,ore=ore,orf=orf,tic=tic)
    else:
        np.savez(file, B=data['B'], t=timesav+data['t'][0], flux=fluxsav, sigma=errsav, fluxorig = data['fluxorig'], errorig = data['errorig'], timeorig = data['timeorig'], mintim = data['mintim'], ort=data['ort'],ore=data['ore'],orf=data['orf'], tic=tic)
    return 0

if __name__ == "__main__":
    matplotlib.use("TkAgg")
    file = 'star_params.npz'
    lc, model, edit = 'lc.data', 'model.txt', 'star_params.npz'
    full, rmpartsctrl, p0ctrl, modelctrl, editctrl = False, False, False, False, False
    ydeg=25,  # degree of the map
    udeg=2,  # degree of the limb darkening
    amp=1,  # amplitude (a value prop. to luminosity)
    dinc, niter = 50, 1
    npts = 4
    sigma, sigmarmf, p0 = 1.9, 0.05, 0
    display, force, noerr, nctrl, rmfctrl = False, False, False, False, False
    if len(argv) < 2: helpf()
    for arg in argv:
        if '-out=' in arg: file = arg.split("=")[-1]
        elif '-tic=' in arg: tic = arg.split("=")[-1]
        elif "-lc=" in arg: lc = arg.split("=")[-1]
        elif "-edit=" in arg: edit, editctrl = arg.split('=')[-1], True 
        elif '-ylm=' in arg: ydeg = str_to_num(arg.split("=")[-1],int,'ylm')
        elif '-sigma=' in arg: sigma = str_to_num(arg.split("=")[-1],float,'sigma')
        elif '-dinc=' in arg: dinc = str_to_num(arg.split("=")[-1],float,'dinc')
        elif '-p0=' in arg: p0, p0ctrl = str_to_num(arg.split("=")[-1],float,'p0'),True
        elif '--rmparts' in arg: rmpartsctrl = True
        elif '--full' in arg: full = True
        elif '--disp' in arg: display = True
        elif '--force' in arg: force = True
        elif '--noerr' in arg: noerr = True
        elif '-rmflares' in arg:
            rmfctrl = True
            if "=" in arg: sigmarmf = str_to_num(arg.split("=")[-1],float,'sigmarmf')
        elif '-niter=' in arg: niter = str_to_num(arg.split("=")[-1],int,'niter')
        elif '-npoints=' in arg: npts, nctrl = str_to_num(arg.split("=")[-1],int,'npts'), True
        elif '-model=' in arg: model, modelctrl = arg.split('=')[-1], True
        elif arg == '-h' or arg == '--help': helpf()

    if os.path.exists(lc) == False and not editctrl:
        print("\a\n# Error! File {} does not exists!\n".format(lc))
        exit()
    elif os.path.exists(edit) == False and editctrl:
        print("\a\n# Error! File {} does not exists!\n".format(edit))
        exit()

    try:
        prepare(tic,file,lc,sigma,ydeg,udeg,amp,dinc,full,display,force,noerr,nctrl,npts,sigmarmf,rmfctrl,niter,rmpartsctrl,p0,p0ctrl,model,modelctrl,edit,editctrl)
    except KeyboardInterrupt:
        print("> Program shut down by user.")