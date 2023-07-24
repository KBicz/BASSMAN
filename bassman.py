#! /usr/bin/python3

import os
import starry
from os.path import exists
from sys import argv, platform

profile = os.getlogin()

if platform != 'darwin': homename = "home"
else: homename = "Users"

if ("--singlecache" not in argv and "-h" not in argv and '--help' not in argv) or not exists(f"/{homename}/{profile}/.theanorc"):
    i = 2
    theano_dir = "/home/{}/.theano/c1/".format(profile)
    while exists(theano_dir): theano_dir, i = "/{}/{}/.theano/c{:d}/".format(homename,profile,i), i+1

    if platform != "darwin": 
        import distro
        if distro.id() == 'debian' or distro.id() == 'mint' or 'pop' in distro.id():
            line = "[global]\ndistro = cpu\nbase_compiledir={}\n\n[blas]\nldflags= -L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lopenblas".format(theano_dir)
        elif distro.id() == 'ubuntu':
            lline = "[global]\ndistro = cpu\nbase_compiledir={}\n\n[blas]\nldflags= -L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lblas".format(theano_dir)
        else:
            line = "[global]\ndistro = cpu\nbase_compiledir={}\n\n[blas]\nldflags= -L/usr/lib64/ -lopenblas".format(theano_dir)
        del distro
    else: line = "[global]\ndistro = cpu\nbase_compiledir={}".format(theano_dir)
    with open("/{}/{}/.theanorc".format(homename,profile),'w') as of: of.write(line)
    del line

del profile

import gc
import sys
import random
import numpy as np
import pymc3 as pm
if starry.__version__ == "1.0.0": import exoplanet as pmx
else: import pymc3_ext as pmx
import matplotlib.pyplot as plt

from os import system
from PIL import Image
from corner import corner
from pandas import DataFrame
from matplotlib import gridspec
from scipy.stats import pearsonr
from IPython.display import display
from matplotlib.figure import Figure
from tqdm import tqdm as progress_bar
from scipy.constants import Stefan_Boltzmann
from matplotlib.ticker import ScalarFormatter
from scipy.ndimage import uniform_filter as smooth
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def helpf():
    print("\n  Program bassman.py for Linux and MacOS written by K. Bicz, ver. of Apr. 26, 2023.")
    print("  Recreating possible locations, sizes and amplitudes of spots on given star.\n")
    print("  Usage: bassman.py [-star=str] [-nspots=int] [-prec=float] [-dprec=float] [-pers=float] [-omb=float]")
    print("         [-obm=float] [-adap=int] [-nstep=int] [-mprod=float] [-finc[=float]] [-gv=float] [-ylm=int]")
    print("         [-ampl=float] [-almin=float] [-almax=float] [-save[=file]] [-amin=float] [-amax=float]")
    print("         [-smin=float] [-smax=float] [-lamin=float] [-lamax=float] [-lomin=float] [-lomax=float]")
    print("         [-gsp=int] [-model=str] [-falpha=float] [-tbjd=float] [-ncores=int] [-nchains=int]")
    print("         [-chi=float] [-logp=float][-temp=float] [-terr=float] [-nth=int] [--pran] [--filter] [--init") 
    print("         [--diff] [--force] [--nodisp] [--sample] [--noadap] [--fulltime] [--nosphere] [--nochi] ")
    print("         [--check] [--ps] [--fscale] [--pcom] [--liveplot] [--modelonly] [--singlecache]")
    print()
    print("         option -star : file with parameters of star (default star = star_params.npz).")
    print("                -nspots : number of spots for starting model (default nspots = 1).")
    print("                -prec : precision for modelling light curve (default prec = 0.000107).")
    print("                -dprec : minimal differential precision (default dprec = 0.00005).")
    print("                -pers : maxminal |1-person| value (default pers = 0.1).")
    print("                -omb : maximal |1-b| value (default omb = 0.015).")
    print("                -obm : interval to fit max model flux (default obm=0.005)")
    print("                -adap : number of successful models to start adaptating (default adap=5).")
    print("                -nstep : number of calculated models to add another spot (default = 30).")
    print("                -mprod : maximal product of correlation to start adapting (default mprod=8).")
    print("                -finc : force exact inclination value (default finc = 70).")
    print("                -gv : value to find if there is a gap in the time series (default gv = 0.2).")
    print("                -ylm : spherical harmonics l number (default ylm = 25).")
    print("                -ampl : maximal amplitude of star (it is by default maximum of model).")
    print("                -almin : minimal alpha for differential rotation modeling (default almin = 0).")
    print("                -almax : maximal alpha for differential rotation modeling (default almax = 1).")
    print("                -save : save modelled stellar spots as image (default image = sspots.eps).")
    print("                -amin : minimal amplitude of the spot (default amin = -0.1).")
    print("                -amax : maximal amplitude of the spot (default amax = 0.0).")
    print("                -smin : minimal surface of stellar spots (default smin = 0.0).")
    print("                -smax : maximal surface of stellar spots (default smax = 0.05).")
    print("                -lamin : minimal latitude of stellar spots (default lamin = -70).")
    print("                -lamax : maximal latitude of stellar spots (default lamax = 70).")
    print("                -lomin : minimal longitude of stellar spots (default lomin = -180).")
    print("                -lomax : maximal longitude of stellar spots (default lomax = 180).")
    print("                -gsp : speed of gif (default gsp = 1).")
    print("                -model : file with starting parameters for spots (default = spots.txt).")
    print("                -falpha : value of differential rotation shear given by user (default falpha = 0.0).")
    print("                -tbjd : make images for selected TBJD (default it is the first time point from data).")
    print("                -ncores : use selected number of cores to sample data (default ncores = 4).")
    print("                -nchains : use selected number of chains to sample data (default nchains = 4).")
    print("                -chi : after reaching selected chi value program accepts calculated model.")
    print("                -logp : after reaching selected logp value program accepts calculated model.")
    print("                -temp : temperature of analysed star (default value is read from TESS_params file).")
    print("                -terr : error of the temperature of analysed star (use only with --pran).")
    print("                -nth : analyze every nth point in data to avoid big RAM usage (default nth = 1).")
    print("                --pran : print analicitcal solutions from Notsu et al. 2019, ApJ, 876:58.")
    print("                --filter : filter output images.")
    print("                --init : plot initial model.")
    print("                --diff : turns on differential rotation.")
    print("                --nodisp : Turns off displaying recreated stellar spots.")
    print("                --force: force to create model for selected nspots.")
    print("                --sample : uses sampling to determine many best models.")
    print("                --noadap : turns off adapting precision to data.")
    print("                --fulltime : reconstruct light curve for whole sector(s).")
    print("                --nosphere : program does not create rotating star sphere gif.")
    print("                --nochi : do not write chi^2 values in legend.")
    print("                --check : do check amplitudes of spots.")
    print("                --ps : print value of scale.")
    print("                --fscale : fit scale of flux to data (default scale = 1/nspots).")
    print("                --pcom : plot light curve of every spot to compare with light curve of star.")
    print("                --liveplot : plot recreate light curve after every spot estimation.")
    print("                --modelonly : save results to file model_results.npz and fisnish program.")
    print("                --singlecache : do not create additional folder for theano cache.")
    print()
    exit()

def save_gif(name,image,time,speed,tic,timedata,fluxdata,fluxmodel,starttbjd,mintim,vmin,vmax,sphere=False,aitoff=False):
    arr = []
    if sphere: 
        nam='sphere'
    elif aitoff:
        nam='aitoff'
        X,Y = np.meshgrid(np.linspace(-np.pi,np.pi,300),np.linspace(-np.pi/2,np.pi/2,300))
    else:
        nam = "map"
    nparts = int(len(timedata)/(image.shape[0]-1))
    for i in progress_bar(range(image.shape[0]),desc="Preparing {} gif".format(nam)):
        arr.append(image[i])
        if sphere:
            fig,ax = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, gridspec_kw={"height_ratios":[2,1]},figsize=(8,9))
            FigureCanvas(fig)
            sto = (i+1)*nparts
            if sto > len(timedata): sto = len(timedata)
            ax[0].set_title("TIC{}".format(tic))
            ax[0].imshow(arr[i],origin="lower",cmap="plasma",vmin=vmin,vmax=vmax)
            ax[0].set_axis_off()
            ax[1].plot(timedata[:sto]-np.min(timedata),fluxdata[:sto],'k.',ms=1,label="Data points")
            ax[1].plot(timedata[:sto]-np.min(timedata),fluxmodel[:sto],'C1',label="Flux model")
            ax[1].set_xlim(np.min(timedata)-np.min(timedata),np.max(timedata)-np.min(timedata))
            ax[1].set_ylim(np.min(fluxdata),np.max(fluxdata))
            ax[1].set_ylabel("Normalized flux",fontsize=15)
            ax[1].set_xlabel("Time [days]",fontsize=15)
            ax[1].set_title("Start TBJD = {}".format(starttbjd))
            ax[1].legend(fontsize=10,numpoints=5)
            plt.close()
        elif aitoff:
            fig = plt.figure(figsize=(12,7))
            FigureCanvas(fig)
            plt.subplot(111,projection='aitoff')
            plt.title("TIC{}\n Time = {:.4f} [days]\n".format(tic,float(time[i]-time[0])),weight='bold')
            plt.pcolormesh(X,Y[::-1],arr[i][::-1],shading='auto',cmap='plasma')
            plt.grid(True)
            plt.xlabel("\nLongitude [deg]",fontsize=15)
            plt.ylabel("Latitude [deg]\n",fontsize=15)
            plt.close()
        elif not sphere and not aitoff:
            fig = Figure()
            FigureCanvas(fig)
            ax = fig.gca()
            ax.set_title("TIC{}\n Time = {:.4f} [days]".format(tic,float(time[i]-time[0])),weight='bold')
            ax.imshow(arr[i],origin="lower",extent=(-180, 180, -90, 90),cmap="plasma")
            ax.set_xlabel('Longitude [deg]')
            ax.set_ylabel('Latitude [deg]')
            ax.margins(0)
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        arr[i] = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        arr[i] = ((arr[i] - arr[i].min()) * (1/(arr[i].max() - arr[i].min()) * 255)).astype('uint8')
        arr[i] = Image.fromarray(arr[i])

    arr[0].save(name,save_all=True,append_images=arr,optimize=False,duration=40*speed,loop=0)  
    if not sphere and not aitoff: plt.close()
    return 0

def print_results(map_soln,nspots,omega_eq,tns):
    tabtab = [" "]
    for i in range(1,nspots+1): tabtab.append("Spot{:d}".format(i))
    data = np.asarray(np.zeros([nspots+1,9]),dtype=str)
    data[0,0], data[0,1] = '-','-'
    data[0,2], data[0,3] = '[deg]','[deg]'
    data[0,4], data[0,5] = '[day]','[K]'
    data[0,6], data[0,7] = '[%]','[R*]'
    data[0,8] = '[W]'
    for i in range(1,nspots+1):
        data[i,0] = "{:.5f}".format(map_soln['amp{:d}'.format(i)])
        data[i,1] = "{:.4f}".format(map_soln['sigma{:d}'.format(i)])
        data[i,2] = "{:.4f}".format(map_soln['lat{:d}'.format(i)])
        data[i,3] = "{:.4f}".format(map_soln['lon{:d}'.format(i)])
        data[i,4] = "{:.4f}".format(360/(omega_eq*(1-map_soln['alpha']*np.sin(map_soln['lat{:d}'.format(i)]*np.pi/180)**2)))
        try: data[i,5] = str(int(round(tns['temp{:d}'.format(i)])))+" ± "+str(int(round(tns['temp{:d}err'.format(i)])))
        except: data[i,5] = "NaN"
        data[i,6] = str(round(tns['psize{:d}'.format(i)],5))
        data[i,7] = "{:.4f}".format(tns['rad{:d}'.format(i)])
        try: data[i,8] = "{:.2e}".format(tns['flux{:d}'.format(i)])
        except: data[i,8] = "NaN"
    print(DataFrame(data,tabtab,["Amplitude","Sigma","Latitude","Longitude","Alpha={:.4f}".format(map_soln['alpha']),"Temperature","Size","Radius","Flux"]))

def init():
    starry.config.lazy = True
    starry.config.quiet = True
    return 0

def savename(file):
    i = 0
    fcopy = file
    while exists(file):
        file = fcopy
        i += 1
        file = file.split(".")
        file = file[0]+"_{:d}.".format(i)+'.'.join(file[1:])
    return file

def str_to_num(string,ntype,name):
    try:
        return ntype(string)
    except ValueError:
        print("\a# Error! {} has to be number!".format(name))
        exit()

def check_value(tab,name):
    if tab[0] >= tab[1]:
        print("\a# Error! Minimal value for {} can't be greater equal maximal!".format(name))
        exit()

def isgap(time,gv):
    for i in range(len(time)-1):
        if time[i+1]-time[i] > gv:
            return True
    return False

def zebra_filter(imag,v1=0.3,v2=0.331,k1=5,k2=2,i1=5,i2=13):
    for _ in range(i1): 
        imag = np.transpose(imag)
        for i in range(len(imag)): 
            imag[i] = smooth(imag[i],k1)
        imag = np.transpose(imag)
    for _ in range(i2): 
        for i in range(len(imag)): 
            imag[i] = smooth(imag[i],k2)
    imag[np.where(imag > v1)] = v2
    for _ in range(i1): 
        imag = np.transpose(imag)
        for i in range(len(imag)): 
            imag[i] = smooth(imag[i],k1)
        imag = np.transpose(imag)
    for _ in range(i2): 
        for i in range(len(imag)): 
            imag[i] = smooth(imag[i],k2)
    return imag

def get_random(start,stop):
    number = 0
    while number == 0:
        number = random.uniform(start,stop)
    return number

def signaltonoise(arr, axis=0, dof=0):
    arr = np.asanyarray(arr)
    m = arr.mean(axis)
    sd = arr.std(axis=axis, ddof=dof)
    return np.where(sd == 0, 0, m/sd)

def tempandsiz(map_soln,B,ampl,nspots):
    tns = {}
    for i in range(1,nspots+1):
        sec = starry.Map(ydeg=B["ydeg"][0], udeg=B["udeg"][0], inc=B['inc'], amp=ampl)
        sec.add_spot(amp=map_soln['amp{:d}'.format(i)]*map_soln['scale'],sigma=map_soln['sigma{:d}'.format(i)],lat=0.0,lon=0.0)
        sec[1] = B["u"][0]
        sec[2] = B["u"][1]
        mapa = sec.render(projection='rect').eval()
        mapa = mapa[::-1]

        lat, lon = 0.0, 00
        laind, loind = 149, 149
        aper, aper2, k, k2 = 0, 0, 1, 1
        if lat < 0: k = -1
        if lon > 0: k2 = -1
        try: 
            if map_soln['amp{:d}'.format(i)] <= 0:
                while mapa[laind+k*aper][loind] < np.median(mapa): aper += 1
            else:
                while mapa[laind+k*aper][loind] > np.median(mapa): aper += 1
        except:
            aper = 150
        try:
            if map_soln['amp{:d}'.format(i)] <= 0:
                while mapa[laind][loind+k2*aper2] < np.median(mapa): aper2 += 1
            else:
                while mapa[laind][loind+k2*aper2] > np.median(mapa): aper2 += 1
        except:
            aper2 = 150
        sta1, sto1 = laind-aper, laind+aper+1
        sta2, sto2 = loind-aper2, loind+aper2+1
        if sta1 < 0 : sta1 = 0
        if sto1 >= 300 : sto1 = 299
        if sta2 < 0 : sta2 = 0
        if sto2 >= 300 : sto2 = 299
        if map_soln['amp{:d}'.format(i)] <= 0:
            msig = np.mean(mapa[sta1:sto1,sta2:sto2][(np.where((mapa[sta1:sto1,sta2:sto2] < np.median(mapa)) & (mapa[sta1:sto1,sta2:sto2] >= 0)))])
            terr = np.std(mapa[sta1:sto1,sta2:sto2][(np.where((mapa[sta1:sto1,sta2:sto2] < np.median(mapa)) & (mapa[sta1:sto1,sta2:sto2] >= 0)))])
        else:
            msig = np.mean(mapa[sta1:sto1,sta2:sto2][(np.where((mapa[sta1:sto1,sta2:sto2] > np.median(mapa)) & (mapa[sta1:sto1,sta2:sto2] >= 0)))])
            terr = np.std(mapa[sta1:sto1,sta2:sto2][(np.where((mapa[sta1:sto1,sta2:sto2] > np.median(mapa)) & (mapa[sta1:sto1,sta2:sto2] >= 0)))])
        tns['psize{:d}'.format(i)] = map_soln['sigma{:d}'.format(i)]*100#(shap/300**2)*100
        tns['rad{:d}'.format(i)] = 2*(tns['psize{:d}'.format(i)]/100)**0.5
        if B['temp'] != 'nan':
            tns['temp{:d}'.format(i)] = msig/np.median(mapa)*float(B['temp'])
            tns['temp{:d}err'.format(i)] = terr/np.median(mapa)*float(B['temp'])
            tns['flux{:d}'.format(i)] = Stefan_Boltzmann*tns['temp{:d}'.format(i)]**4*tns['psize{:d}'.format(i)]/100
            tns['flux{:d}err'.format(i)] = Stefan_Boltzmann*tns['temp{:d}err'.format(i)]**4*tns['psize{:d}'.format(i)]/100
        else:
            tns['temp{:d}'.format(i)] = 'nan'
            tns['temp{:d}err'.format(i)] = 'nan'
            tns['flux{:d}'.format(i)] = 'nan'
            tns['flux{:d}err'.format(i)] = 'nan'
    return tns

def analytical_model(temp,ampl,minf):
    if str(temp).lower() != 'nan':
        tspot = 0.751*temp - 3.58e-5*temp**2 + 808
        aspot = 100*(1-minf/ampl)*(1-(tspot/temp)**4)**(-1)
        return tspot, aspot
    else:
        return 0.0, 0.0

def read_model(file):
    if not exists(file):
        print("# Error! Model file {} does not exists!".format(file))
        exit()
    with open(file,'r') as of: lines = of.readlines()
    amplw, sigms, lats, lons, alpha, scale = [], [], [], [], 0.0, 0.0
    for i in range(len(lines)):
        line = lines[i].split("\n")[0].split()
        try:
            if "Spot" in line[0]:
                amplw.append(float(line[1]))
                sigms.append(float(line[9])/100.)
                lats.append(float(line[3]))
                lons.append(float(line[4]))
            elif "scale" == line[1]:
                scale = float(line[3])
            if "#" not in line:
                if "Alpha=" in line[5]:
                    alpha = float(line[5].split("=")[1])
                elif "Alpha=" in line[4]:
                    alpha = float(line[4].split("=")[1])
        except:
            pass
    try: return np.array(amplw), np.array(sigms), np.array(lats), np.array(lons), alpha, scale
    except: return np.array(amplw), np.array(sigms), np.array(lats), np.array(lons), alpha, 1/len(amplw)

def checkmodel(amps,sizs,lats,lons,alpha,ampran,sizran,latran,lonran,alphatab,nspots):
    values, ranges = [amps,sizs,lats,lons], [ampran,sizran,latran,lonran]
    for val, rang, i in zip(values,ranges,["amplitudes","sizes","latitutdes",'longitudes']):
        if np.max(val) > np.max(rang) or np.min(val) < np.min(rang) or alpha > np.max(alphatab) or alpha < np.min(alphatab):
            print("\n\a# Error! One of the {} is not in the range of given parameters!".format(i))
            exit()

def recreate_sspots(params,nspots,prec,dprec,prs,omb,obm,def_inclination,gv,ylm,stara,alphatab,alu,slu,lalu,lolu,adap,mprod,savectrl,save,force,displayctrl,sample,noadap,finc,gsp,nstep,diffctrl,filterctrl,fulltime,nosphere,command,modelfile,modelctrl,staractrl,tbjd,tbjdctrl,initctrl,ncores,nchains,chiv,logpv,chivctrl,logpvctrl,nochi,temper,temperctrl,check,pcom,ps,fscale,pran,falpha,falphactrl,publication,liveplot,terrctrl,terr,modelonly,nth):

    if publication:
        legpara = {'size':15,'weight':'bold'}
        markersize, msc = 4, 2
        yts, ytw = 25, 'normal'
        yls, ylw = 28, 'bold'
        xts, xtw = 25, 'normal'
        xls, xlw = 28, 'bold'
        ylp, xlp = 12, 12 
    else: 
        legpara = {'size':10}
        markersize, msc = 2, 1
        yts, ytw = 10, 'normal'
        yls, ylw = 20, 'normal'
        xts, xtw = 10, 'normal'
        xls, xlw = 20, 'normal'
        ylp, xlp = 4, 4

    if liveplot: plt.ion()
    cnums = [0,1,2,4,5,6,7,9,8]
    data = np.load(params, allow_pickle=True)
    B = data["B"].item()
    t = data["t"][::nth]
    tbjdnam = str(t[0]+data['mintim'])
    if tbjdctrl:
        tbjdnam = str(tbjd)
        if (tbjd < t[0] + data['mintim'] or tbjd > t[-1] + data['mintim']): 
            print("% Warning! Given TBJD = {} is not in observational interval {}-{}.".format(tbjd,t[0] + data['mintim'],t[-1] + data['mintim']))
            while tbjd < t[0] + data['mintim']: tbjd += B['prot']
            while tbjd > t[-1] + data['mintim']: tbjd -= B['prot']
        tbjd -= (data['mintim']+t[0])
    else: tbjd = t[0]-t[0]

    if diffctrl: t_ani = np.linspace(tbjd,tbjd+3*B['prot'],100)
    else: t_ani = np.linspace(tbjd,tbjd+B['prot'],100)

    flux = data["flux"][::nth]
    sigma = data['sigma'][::nth]
    if not staractrl: ampl = np.mean(flux)+1.4*np.std(flux)
    else: ampl = stara
    omega_eq, addon = 360/B['prot'], '_diff'
    if not diffctrl: alphaa, alpha, addon = 0.0, 0.0, '' # differential rotation shear
    if not diffctrl and falphactrl: alphaa, alpha, addon = falpha, falpha, '_diff'
    if finc: B['inc'] = def_inclination
    B["ydeg"] = (ylm,)
    try: tic = data['tic']
    except: tic = ""
    if temperctrl: B['temp'] = temper

    ar = (alu[0],alu[1]) 
    sr = (slu[0],slu[1]) 
    lar = (lalu[0],lalu[1])
    lor = (lolu[0],lolu[1]) 
    spots_bools = [True for _ in range(nspots)]
    m, bbb, mf, pers, iterat, chisqselnc, logp, amplctrl = 5, 0.000, -200.0, 0, 0, 1000, -100000000., False
    precd, logp, ms, chisqtab, scalemc = prec, 0, [], [], 1.0/nspots
    amptab, sigmatab, lattab, lontab, alphaatab = [], [], [], [], []

    if modelctrl:
        amps,sigms,lats,lons,alphaa,scalemc = read_model(modelfile)
        ampso,sigmso,latso,lonso = amps.copy(),sigms.copy(),lats.copy(),lons.copy()
        checkmodel(ampso,sigmso,latso,lonso,alphaa,ar,sr,lar,lor,alphatab,nspots)
        if not diffctrl: alpha = 0.0
        if not diffctrl and falphactrl: alpha = falpha

    while (m > prec or abs(1-round(bbb,3)) > omb or abs(1-round(pers,2)) > prs or mf < ampl-obm or mf > ampl+obm or amplctrl == False):
        gc.collect()
        print()
        if iterat > nstep and not force:
            iterat = 0
            nspots += 1
            prec, ms, chisqtab = precd, [], []
            print("\a% Model for {} spots does not converge, increasing number of spots to {}.".format(nspots-1,nspots))

        iterat += 1
        if ((iterat > nstep+20 and not diffctrl) or (iterat > 2*(nstep+20) and diffctrl)) and force:
            print("\a# Error! Model for {} spots will not converge!".format(nspots))
            if len(chisqtab) > 0:
                print("> Recreating model with best chi^2.")
                whchisq = int(np.array(chisqtab).argmin())
                amps,sigms,lats,lons,alphaa = amptab[whchisq], sigmatab[whchisq], lattab[whchisq], lontab[whchisq], alphaatab[whchisq]
                modelctrl = True
            else:
                exit()

        if m == 1000000*precd: print("# Error! Infinite model!")
        else:
            if ((len(ms) >= adap and abs(m-prec) >= dprec and round(min(ms)/prec) <= mprod and not noadap) and ((prec == precd) or (prec != precd and min(ms)+dprec < prec))) and not logpvctrl and not chivctrl:
                prec = min(ms)+dprec
                ms = list(np.array(ms)[np.where(np.array(ms) > min(ms))])
                print("\a% Starting adaptative mode. New precision = {}.".format(prec))
            elif iterat > 1:
                print("# Error! Desired precision has not been achieved!")
                amplctrl = False
                if m > prec: print("# Estimation = {} but has to be less equal {}.".format(m,prec))
                if abs(1-round(bbb,3)) > omb: print("# |1-b| = {} but has to be less equal {}.".format(abs(1-round(bbb,3)),omb))
                if abs(1-round(pers,2)) > prs: print("# |1-pearson| = {} but has to be less equal {}.".format(abs(1-round(pers,2)),prs))
                if mf < ampl-obm or mf > ampl+obm: print("# Flux_model_max = {} but has to be between {}-{}.".format(mf,ampl-obm,ampl+obm))
                if False in spots_bools:
                    for itera in range(len(spots_bools)):
                        if spots_bools[itera] == False:
                            print("# Amplitude of spot {:d} is smaller than signal of star!".format(itera+1))
                    spots_bools = [True for _ in range(nspots)]
            if logpvctrl and logpv > logp and iterat > 2: print("# Error! Desired logp has not been achieved! logp has to be greater than {}.".format(logpv))
            if chivctrl and chiv < chisqselnc and iterat > 2: print("# Error! Desired chisq has not been achieved! chisq has to be less than {}.".format(chiv))

        gc.collect()
        print("> Estimating model no. {} for {} spots.".format(iterat,nspots))
        with pm.Model() as model:
            sec = [None for _ in range(nspots)]
            omega = [None for _ in range(nspots)]
            flux_model = [None for _ in range(nspots)]

            scalealpha, scalealphanames = [], []
            if fscale:
                scale = pm.Uniform("scale", lower=0, upper=1.0, testval=scalemc)
                scalealpha.append(scale)
                scalealphanames.append("scale")
            else:
                scale = 1./nspots
            if diffctrl: 
                if not modelctrl: alphaa = random.uniform(alphatab[0],alphatab[1])
                alpha = pm.Uniform("alpha", lower=alphatab[0], upper=alphatab[1], testval=alphaa)
                scalealpha.append(alpha)
                scalealphanames.append("alpha")

            sspots = {}
            if not modelctrl:
                amps = [get_random(ar[0],ar[1]) for _ in range(nspots)]
                sigms = [get_random(sr[0],sr[1]) for _ in range(nspots)]
                lats = [get_random(lar[0],lar[1]) for _ in range(nspots)]
                lons = [get_random(lor[0],lor[1]) for _ in range(nspots)]
            elif modelctrl and nspots > len(amps):
                amps, sigms, lats, lons = list(ampso.copy()), list(sigmso.copy()), list(latso.copy()), list(lonso.copy())
                for _ in range(nspots-len(amps)):
                    amps.append(get_random(ar[0],ar[1]))
                    sigms.append(get_random(sr[0],sr[1]))
                    lats.append(get_random(lar[0],lar[1]))
                    lons.append(get_random(lor[0],lor[1]))
                amps, sigms, lats, lons = np.array(amps), np.array(sigms), np.array(lats), np.array(lons)
            elif modelctrl and iterat > 1:
                if iterat < 2: print("\n# Spot model with given model parameters does not converge! BASSMAN turns on normal mode.")
                amps = [get_random(ar[0],ar[1]) for _ in range(nspots)]
                sigms = [get_random(sr[0],sr[1]) for _ in range(nspots)]
                lats = [get_random(lar[0],lar[1]) for _ in range(nspots)]
                lons = [get_random(lor[0],lor[1]) for _ in range(nspots)]
            for i in range(1,nspots+1):
                sspots['amp{:d}'.format(i)] = pm.Uniform("amp{:d}".format(i), lower=alu[0], upper=alu[1], testval=amps[i-1])
                sspots['sigma{:d}'.format(i)] = pm.Uniform("sigma{:d}".format(i), lower=slu[0], upper=slu[1], testval=sigms[i-1])
                sspots['lat{:d}'.format(i)] = pm.Uniform("lat{:d}".format(i), lower=lalu[0], upper=lalu[1], testval=lats[i-1])
                sspots['lon{:d}'.format(i)] = pm.Uniform("lon{:d}".format(i), lower=lolu[0], upper=lolu[1], testval=lons[i-1])

                sec[i-1] = starry.Map(ydeg=B["ydeg"][0], udeg=B["udeg"][0], inc=B['inc'], amp=ampl)
                sec[i-1].add_spot(amp=sspots['amp{:d}'.format(i)],sigma=sspots['sigma{:d}'.format(i)],lat=sspots['lat{:d}'.format(i)],lon=sspots['lon{:d}'.format(i)])
                sec[i-1][1] = B["u"][0]
                sec[i-1][2] = B["u"][1]

                # Compute the flux model
                omega[i-1] = omega_eq*(1-alpha*np.sin(sspots['lat{:d}'.format(i)]*np.pi/180)**2)
                flux_model[i-1] = sec[i-1].flux(theta=omega[i-1] * t)
                gc.collect()
                
            flux_model = scale*pm.math.sum(flux_model,axis=0)
            gc.collect()
            pm.Deterministic("flux_model", flux_model)
            # Save our initial guess
            gc.collect()
            flux_model_guess = pmx.eval_in_model(flux_model).copy()
            gc.collect()
            pm.Normal("obs", mu=flux_model, sd=sigma, observed=flux)
            gc.collect()
            if starry.__version__ == "1.0.0": map_soln = pmx.optimize(vars=[*sspots.values(),*scalealpha],options={"maxiter": 10000},start=model.test_point)
            else: map_soln = pmx.optimize(vars=[*sspots.values(),*scalealpha],maxeval=10000,start=model.test_point)
            gc.collect()
            if str(map_soln['flux_model'][0]) != "nan":
                for i in range(1,nspots+1):
                    if starry.__version__ == "1.0.0": map_soln = pmx.optimize(vars=[sspots['amp{:d}'.format(i)],sspots['sigma{:d}'.format(i)],sspots['lat{:d}'.format(i)],sspots['lon{:d}'.format(i)],*scalealpha],start=map_soln,options={"maxiter": 10000})
                    else: map_soln = pmx.optimize(vars=[sspots['amp{:d}'.format(i)],sspots['sigma{:d}'.format(i)],sspots['lat{:d}'.format(i)],sspots['lon{:d}'.format(i)],*scalealpha],start=map_soln,maxeval=10000)
                    gc.collect()
                if starry.__version__ == "1.0.0": map_soln, info = pmx.optimize(vars=[*sspots.values(),*scalealpha], start=map_soln, return_info=True, options={"maxiter": 10000})
                else: map_soln, info = pmx.optimize(vars=[*sspots.values(),*scalealpha], start=map_soln, return_info=True, maxeval=10000)
                gc.collect()
                logp = -info.fun
            else:
                m = 'nan'
        
        if not fscale: 
            map_soln['scale'] = scale
            nc = nspots*4
        else:
            nc = nspots*4+1
        if not diffctrl: map_soln['alpha'] = alpha
        else: nc += 1
        fmchi = pmx.eval_in_model(flux_model, map_soln, model=model)
        m, bbb = abs(np.polyfit(t, flux/fmchi, 1))
        mf, chisqsel, chisqselnc = max(fmchi), np.sum((flux-fmchi)**2/(sigma)**2)/len(fmchi), np.sum((flux-fmchi)**2/(sigma)**2)/(len(fmchi)-nc)
        print("chi^2/(N-{:d}) = {}".format(nc,chisqselnc))
        if ps: print("scale = {}".format(map_soln['scale']))
        if (str(m) != 'nan' and abs(1-round(pers,2)) <= prs or mf >= ampl-obm or mf <= ampl+obm): ms.append(m)

        if str(m) == 'nan' or str(map_soln['flux_model'][0]) == "nan":
            m, bbb, mf, pers = 1000000*precd, 0.0, -200.0, 0.0
        else:
            pers = pearsonr(flux,fmchi)[0]

        if str(chisqsel).lower() != 'nan':
            amptab.append(amps)
            sigmatab.append(sigms)
            lattab.append(lats)
            lontab.append(lons)
            alphaatab.append(alphaa)

        gc.collect()
        if liveplot:
            if 'figur' not in globals(): figur = plt.figure(111,figsize=(10, 6))
            plt.clf()
            plt.plot(t+data['mintim'],flux, "k.", ms=2, label="Data")
            plt.pause(0.00000001)
            plt.plot(t+data['mintim'],fmchi,'C1.',label="Light curve model",ms=2)
            plt.pause(0.00000001)
            plt.plot([],[],' ',label="logp = {}".format(logp))
            plt.pause(0.00000001)
            plt.plot([],[],' ',label=r"$\chi^2_{\,\,sel}$/(N-"+str(nc)+r") = {}".format(chisqselnc))
            plt.pause(0.00000001)
            plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            plt.legend(numpoints=5)
            plt.xlabel("TBJD [days]", fontsize=15, weight='bold', labelpad=xlp)
            plt.ylabel("Normalized flux", fontsize=15, weight='bold', labelpad=ylp)
            plt.pause(0.00000001)
            plt.show()
            gc.collect()
        
        if check:
            with model:
                teststar = [None for _ in range(nspots)]
                testomega = [None for _ in range(nspots)]
                spots_bools = [True for _ in range(nspots)]
                flux_model_comp = pmx.eval_in_model(flux_model, map_soln, model=model)
                for n in range(1,nspots+1):
                    teststar[n-1] = starry.Map(ydeg=B["ydeg"][0], udeg=B["udeg"][0], inc=B['inc'], amp=ampl)
                    teststar[n-1].add_spot(amp=map_soln['amp{:d}'.format(n)]*map_soln['scale'],sigma=map_soln['sigma{:d}'.format(n)],lat=map_soln['lat{:d}'.format(n)],lon=map_soln['lon{:d}'.format(n)])
                    teststar[n-1][1] = B["u"][0]
                    teststar[n-1][2] = B["u"][1]
        
                    testomega[n-1] = omega_eq*(1-map_soln['alpha']*np.sin(sspots['lat{:d}'.format(n)]*np.pi/180)**2)
                    flux_model_test = pmx.eval_in_model(teststar[n-1].flux(theta=testomega[n-1] * t), map_soln, model=model)
                    flux_model_test += ampl-np.max(flux_model_test)
                    if pcom:
                        plt.figure(figsize=(9, 7))
                        plt.plot(t,flux_model_test,label="Light curve of spot {:d}".format(n))
                        plt.plot(t,flux_model_comp,label="Light curve model")
                        plt.title("Spot {:d}".format(n),weight="bold")
                        plt.xlabel("TBJD [days]",fontsize=15)
                        plt.ylabel("Normalized flux",fontsize=15)
                        plt.legend()
                        plt.show()
                    if len(np.where(flux_model_test-flux_model_comp < -1e-5)[0]) > 0: spots_bools[n-1] = False
                if False not in spots_bools: amplctrl = True
                gc.collect()
        else: 
            amplctrl = True
            gc.collect()
        
        if ((((iterat > nstep+20 and not diffctrl) or (iterat > 2*(nstep+20) and diffctrl)) and force) or (logp >= logpv and logpvctrl) or (chisqselnc <= chiv and chivctrl)) and amplctrl == True: break
        if (logp < logpv and logpvctrl) or (chisqselnc > chiv and chivctrl): pers = 0.0

    if liveplot: 
        plt.close(111)
        plt.ioff()
    print("\n"+"\033[1m"+"> Success! Desired precision achieved!"+"\033[0m"+"\n")
    
    if staractrl: mf = stara
    for i in range(1,nspots+1):
        if map_soln["sigma{:d}".format(i)] >= 0.5: print("\a% Warning! Size of spot no. {:d} is very big! Model may not be correct!".format(i))
        elif map_soln["sigma{:d}".format(i)] <= 0.001: print("\a% Warning! Size of spot no. {:d} is very small! Model may not be correct!".format(i))
    
    tns=tempandsiz(map_soln,B,ampl,nspots)
    print_results(map_soln,nspots,omega_eq,tns)
    print()
    gc.collect()

    if not diffctrl: whfl = np.where( ((data['timeorig']-data['mintim'] >= t[0]) & (data['timeorig']-data['mintim'] <= t[0]+B['prot'])) )[0]
    else: whfl = np.where( ((data['timeorig']-data['mintim'] >= t[0]) & (data['timeorig']-data['mintim'] <= t[0]+3*B['prot'])) )[0]

    fluxsum = 0
    for i in range(1,nspots+1):
        try: fluxsum += tns['flux{:d}'.format(i)]
        except: pass

    if modelonly:
        np.savez(savename("model_results.npz"),B=B,map_soln=map_soln,nspots=nspots,nc = 4*nspots,addon=addon,omega_eq = 360/B['prot'],fulltime=fulltime,gv=gv,ampl=ampl,publication=publication,amplctrl=amplctrl,terrctrl=terrctrl,terr=terr,tns=tns,displayctrl=displayctrl,tic=tic,nochi=nochi,chisqselnc=chisqselnc,fmchi=fmchi,command=command,mf=mf,logp=logp,fluxsum=fluxsum,pers=pers,m=m,bbb=bbb,whfl=whfl,t_ani=t_ani,tbjdctrl=tbjdctrl,gsp=gsp,filterctrl=filterctrl,tbjd=tbjd,tbjdnam=tbjdnam)
        exit()
    
    secimshow = starry.Map(ydeg=B["ydeg"][0], udeg=B["udeg"][0], inc=B['inc'], amp=ampl, alpha=map_soln['alpha'])
    for i in range(1,nspots+1):
        secimshow.add_spot(amp=map_soln['amp{:d}'.format(i)]*map_soln['scale'],sigma=map_soln['sigma{:d}'.format(i)],lat=map_soln['lat{:d}'.format(i)],lon=map_soln['lon{:d}'.format(i)])
    secimshow[1] = B["u"][0]
    secimshow[2] = B["u"][1]
    gc.collect()

    with model:
        res=300
        image = np.zeros((len(t_ani),res,res))
        imagefil = np.zeros((len(t_ani),res,res))
        for n in range(nspots):
            tmp = pmx.eval_in_model(sec[n].render(projection="rect",res=res),point=map_soln)
            shift = np.array(pmx.eval_in_model(omega[n],point=map_soln)*t_ani*res/360,dtype=int)
            for n in range(len(t_ani)):
                image[n] += np.roll(tmp,shift[n],axis=1)

        gc.collect()
        imageshow = np.zeros([res,res])
        if not tbjdctrl: #imageshow = secimshow.render(theta=(t[0]/B['prot']-t[0]//B['prot'])*360,projection="rect").eval()
            for n in range(nspots):
                tmp = pmx.eval_in_model(sec[n].render(projection="rect",res=res),point=map_soln)
                shift = np.array(pmx.eval_in_model(omega[n],point=map_soln)*(t[0]-t[0])*res/360,dtype=int)
                imageshow += np.roll(tmp,shift,axis=1)
        else: #imageshow = secimshow.render(theta=(tbjd/B['prot']-tbjd//B['prot'])*360,projection="rect").eval()
            for n in range(nspots):
                tmp = pmx.eval_in_model(sec[n].render(projection="rect",res=res),point=map_soln)
                shift = np.array(pmx.eval_in_model(omega[n],point=map_soln)*tbjd*res/360,dtype=int)
                imageshow += np.roll(tmp,shift,axis=1)

        if filterctrl: imageshow = zebra_filter(imageshow)

        gc.collect()
        if displayctrl or savectrl:
            #Filtering images if needed
            if filterctrl:
                for i in progress_bar(range(len(t_ani)),desc="Filtering images"):
                    imagefil[i] = zebra_filter(image[i])
            else:
                imagefil = image.copy()

    gc.collect()
    save2 = save.split(".")
    save2 = "{}_selection_flux.{}".format(save2[0],save2[1])
    if savectrl and not exists(save2):
        plt.figure(1)
        plt.plot(data['timeorig'], data['fluxorig']/1000+1, "C0.",ms=1)
        plt.plot(t+data['mintim'],flux,'C1.',ms=1)
        plt.xlabel("TBJD [days]")
        plt.ylabel("Flux")
        plt.gca().get_xaxis().set_major_formatter(ScalarFormatter(useOffset=False))
        plt.gca().get_yaxis().set_major_formatter(ScalarFormatter(useOffset=False))
        plt.savefig(save2)
        plt.close(1)
     
    gc.collect()
    #recreating model and missing data
    flux_model2 = [None for _ in range(nspots)]
    sec = [None for _ in range(nspots)]
    flux_model3 = [None for _ in range(nspots)]
    omega = [None for _ in range(nspots)]
    ttt = data['timeorig']-data['mintim']
    if fulltime: ttt2 = data['ort']
    if isgap(t,gv):
        t2 = np.arange(t[0],t[-1],abs(t[1]-t[0]))
        flux_model2 = [None for _ in range(nspots)]
    elif isgap(data['ort'],gv) and fulltime:
        t2 = np.arange(data['ort'][0],data['ort'][-1],abs(t[1]-t[0]))
    flux_model4 = [None for _ in range(nspots)]
    for i in range(1,nspots+1):
        sec[i-1] = starry.Map(ydeg=B["ydeg"][0], udeg=B["udeg"][0], inc=B['inc'], amp=ampl)
        sec[i-1].add_spot(amp=map_soln['amp{:d}'.format(i)],sigma=map_soln['sigma{:d}'.format(i)],lat=map_soln['lat{:d}'.format(i)],lon=map_soln['lon{:d}'.format(i)])
        sec[i-1][1] = B["u"][0]
        sec[i-1][2] = B["u"][1]
        
        omega[i-1] = omega_eq*(1-map_soln['alpha']*np.sin(sspots['lat{:d}'.format(i)]*np.pi/180)**2)
        if fulltime: flux_model4[i-1] =  sec[i-1].flux(theta=omega[i-1] * ttt2)
        if isgap(t,gv) or (isgap(data['ort'],gv) and fulltime): flux_model2[i-1] = sec[i-1].flux(theta=omega[i-1] * t2)
        flux_model3[i-1] = sec[i-1].flux(theta=omega[i-1] * ttt)
        gc.collect()

    if isgap(t,gv) or (isgap(data['ort'],gv) and fulltime): flux_model2 = map_soln['scale']*pm.math.sum(flux_model2,axis=0)
    flux_model3 = map_soln['scale']*pm.math.sum(flux_model3,axis=0)
    flux_model_orig = pmx.eval_in_model(flux_model3, map_soln, model=model)
    del flux_model3
    if fulltime:
        flux_model4 = map_soln['scale']*pm.math.sum(flux_model4,axis=0)
        flux_model4 = pmx.eval_in_model(flux_model4, map_soln, model=model)
    chisq = np.sum(((data['fluxorig']/1000+1)-flux_model_orig)**2/(data['errorig']/1000)**2)
    if fulltime: chisqfull = np.sum((data['orf']-flux_model4)**2/(data['ore'])**2)

    # Analytical and estimated temperatures and areas of spots
    fs, aspots, terrmean = 0, 0, 0
    for i in range(1,nspots+1): 
        try:    
            fs += tns['flux{:d}'.format(i)]
            terrmean += tns['flux{:d}err'.format(i)]**2
        except:
            fs, terrmean = 0.0, 0.0
        aspots += tns['psize{:d}'.format(i)]
        gc.collect()
    atemp, aspoteddnes = analytical_model(float(B['temp']),ampl,min(flux_model_orig))
    if pran:
        if amplctrl: print("% Assumed star amplitude = {}".format(ampl))
        else: print("% Estimated star amplitude = {}".format(ampl))
        if terrctrl:
            terror = terr*(0.751-0.0000716*float(B['temp']))
            serror = np.sqrt((400*(1-min(flux_model_orig)/ampl)*atemp**3/(float(B['temp'])**4*(1-(atemp/float(B['temp']))**4)**2)*terror)**2 + (400*(1-min(flux_model_orig)/ampl)*atemp**4/(float(B['temp'])**5*(1-(atemp/float(B['temp']))**4)**2)*terr)**2)
            print("% Analytical spotedness of star = {:.2f} ± {:.2f} %".format(aspoteddnes,serror))
            print("% Analytical mean temperature of spots = {:.0f} ± {:.0f} K".format(atemp,terror))
            print("% Estimated spotedness of star = {:.2f} %".format(aspots))
            print("% Estimated mean temperature of spots = {:.0f} ± {:.0f} K\n".format((fs/(aspots*0.01*Stefan_Boltzmann))**0.25,(np.sqrt(terrmean)/(aspots*0.01*Stefan_Boltzmann))**0.25))
        else:
            print("% Analytical spotedness of star = {:.2f} %".format(aspoteddnes))
            print("% Analytical mean temperature of spots = {:.0f} K".format(atemp))
            print("% Estimated spotedness of star = {:.2f} %".format(aspots))
            print("% Estimated mean temperature of spots = {:.0f} K\n".format((fs/(aspots*0.01*Stefan_Boltzmann))**0.25))

    gc.collect()
    components, maxcom = {}, []
    if fulltime: tc = data['ort']
    else: tc = ttt
    with model:
        for i in range(1,nspots+1):
            components["Spot {:d}".format(i)] = pmx.eval_in_model(sec[i-1].flux(theta=omega[i-1]*tc),map_soln, model=model)*map_soln['scale']
            if map_soln['amp{:d}'.format(i)] < 0: maxcom.append(np.max(components["Spot {:d}".format(i)]))
            else: maxcom.append(np.min(components["Spot {:d}".format(i)]))
            gc.collect()
    if len(maxcom) == 0: maxcom = map_soln['scale']*mf
    #else: maxcom = np.max(maxcom)
    
    gc.collect()
    if savectrl:
        save2 = save.split(".")
        if not fulltime: sfil = savename("{}_components_inc{:d}_n{:d}{}.txt".format(save2[0],int(B['inc']),nspots,addon))
        else: sfil = savename("{}_components_all_inc{:d}_n{:d}{}.txt".format(save2[0],int(B['inc']),nspots,addon))
        _, cmps = zip(*components.items())
        np.savetxt(sfil,np.transpose([tc+data['mintim'],*cmps]))
        del cmps
        gc.collect() 

    # Saving data to files
    if savectrl:
        save2 = save.split(".")
        sfil = savename("{}_spots_inc{:d}_n{:d}{}.txt".format(save2[0],int(B['inc']),nspots,addon))
        default_sysout = sys.stdout
        sys.stdout = open(sfil,'w')
        print("# {}".format(command))
        print_results(map_soln,nspots,omega_eq,tns)
        print("# Flux_sum = {:.3e} W".format(fs))
        print("# Estimated mean temperature of spots = {:.0f} K".format((fs/(aspots*0.01*Stefan_Boltzmann))**0.25))
        print("# Estimated spotedness of star = {:.2f} %".format(aspots))
        if not terrctrl:
            print("# Analytical mean temperature of spots = {:.0f} K".format(atemp))
            print("# Analytical spotedness of star = {:.2f} %".format(aspoteddnes))
        else:
            print("# Analytical spotedness of star = {:.2f} ± {:.2f} %".format(aspoteddnes,serror))
            print("# Analytical mean temperature of spots = {:.0f} ± {:.0f} K".format(atemp,terror))
        print("# chi^2/(N-{:d}) = {}".format(nc,chisqselnc))
        print("# std_dev = {:.2f}*sigma".format(np.max(np.abs(flux-fmchi))/np.std(flux-fmchi)))
        print("# SNR = {}".format(signaltonoise(flux-fmchi+1)))
        print("# scale = {}".format(map_soln['scale']))
        sys.stdout.close()
        sys.stdout = default_sysout
        gc.collect()

        sfil = savename("{}_flux_inc{:d}_n{:d}{}.txt".format(save2[0],int(B['inc']),nspots,addon))
        if not fulltime:
            np.savetxt(sfil, np.transpose([data['timeorig'],data['fluxorig'], data['errorig'],(flux_model_orig-1.)*1000]))
        else:
            np.savetxt(sfil, np.transpose([data['ort']+data['mintim'],(data['orf']-1.)*1000, data['ore']*1000,(flux_model4-1.)*1000]))
        
        gc.collect()
        sfil = savename("{}_trend_inc{:d}_n{:d}{}.txt".format(save2[0],int(B['inc']),nspots,addon))
        if not fulltime:
            np.savetxt(sfil, np.transpose([data['timeorig'],data['fluxorig']-((flux_model_orig-1.)*1000), data['errorig'],np.ones(len(data['timeorig']))*2137]))
        else:
            np.savetxt(sfil, np.transpose([data['ort']+data['mintim'],((data['orf']-1.)*1000)-((flux_model4-1.)*1000),data['ore']*1000,np.ones(len(data['ort']))*2137]))

        gc.collect()
        if fulltime: fnamelogp = "{}_logp_all_n{:d}{}.txt".format(save2[0],nspots,addon)
        else: fnamelogp = "{}_logp_n{:d}{}.txt".format(save2[0],nspots,addon)
        if exists(fnamelogp): mode = 'a'
        else: mode = 'w'
        with open(fnamelogp,mode) as of:
            if mode == "w":
                string = "% Inclination_[deg]     logp       max_flux    chisq_sel    chisq_flare    "
                if fulltime:
                    string += "chisq_full    "
                string += "flux_sum    |1-pers|        a        b     "
                for nsp in range(1,nspots+1):
                    string += "   T{:d}   Terr{:d}      S{:d}   ".format(nsp,nsp,nsp)
                of.write(string)
            string = "\n        {:d}          {:.5f}   {:.5f}     {:.5f}     {:.5f}      ".format(int(B['inc']),logp,mf,chisqselnc,chisq/(len(data['timeorig'])-nc))
            if fulltime: string += "{:.5f}      ".format(chisqfull/(len(data['orf'])-nc))
            string += "{:.2e}    {:.7f}   {:.8f}  {:.5f}".format(fluxsum,abs(1-pers),m,bbb)
            for nsp in range(1,nspots+1):
                try:
                    string += "   {:d}   {:d}   {:.5f}".format(int(round(tns['temp{:d}'.format(nsp)])),int(round(tns['temp{:d}err'.format(nsp)])),tns['psize{:d}'.format(nsp)])
                except:
                    string += "   NaN   NaN   {:.5f}".format(tns['psize{:d}'.format(nsp)])
            of.write(string)
        gc.collect()
        if mode == 'a':
            with open(fnamelogp,'r') as of:
                logplines = of.readlines()
            logpheader = logplines[0]
            logplines = sorted(logplines[1:])
            with open(fnamelogp,'w') as of:
                of.write(logpheader.split("\n")[0])
                for line in logplines: 
                    try: of.write("\n{}".format(line.split("\n")[0]))
                    except: pass

    gc.collect()
    # Creating images/plots
    if displayctrl or savectrl:
        plt.figure(1,figsize=(13, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1],sharex=ax0)
        plt.subplots_adjust(hspace=0,top=0.933,bottom=0.082)
        if fulltime:
            ax0.plot(data['ort']+data['mintim'], data['orf'], "k.", ms=2, label="Data")
            lll = "Model for selected interval"
            ax1.plot(data['ort']+data['mintim'], data['orf']-flux_model4,'k.', ms=2)
            ax1.plot([data['ort'][0]+data['mintim'],data['ort'][-1]+data['mintim']],[0,0],'C4--')
        else:
            ax0.plot(data['timeorig'], data['fluxorig']/1000+1, "k.", ms=2, label="Data")
            ax1.plot(data['timeorig'], (data['fluxorig']/1000+1)-flux_model_orig, "k.", ms=2)
            ax1.plot([data['timeorig'][0],data['timeorig'][-1]],[0,0],'C2--')
            lll = "Model"
        if isgap(t,gv) or (isgap(data['ort'],gv) and fulltime): ax0.plot(t2+data['mintim'], pmx.eval_in_model(flux_model2,map_soln, model=model),marker='.',color="limegreen",label="Missing",ms=1,linestyle='')
        if fulltime: ax0.plot(data['ort']+data['mintim'],flux_model4,'C9.',label="Light curve model",ms=1,linestyle='')
        ax0.plot(data['timeorig'], flux_model_orig, "C1.", label=lll,ms=1,linestyle='')
        if initctrl: ax0.plot(t+data['mintim'], flux_model_guess,marker='.',color="dodgerblue",alpha=0.6, label="Initial model",ms=1,linestyle='')
        if not nochi:
            ax0.plot([],[],' ',label=r"$\chi^2_{\,\,sel}$/(N-"+str(nc)+r") = {}".format(chisqselnc))
            ax0.plot([],[],' ',label=r"$\chi^2_{\,\,flare}$"+r"/(N-{:d}) = {}".format(nc,chisq/(len(data['timeorig'])-nc)))
            if fulltime:
                ax0.plot([],[],' ',label=r"$\chi^2_{\,\,full}$/(N-"+str(nc)+r") = "+str(chisqfull/(len(data['orf'])-nc)))
        plt.setp(ax0.get_xticklabels(), visible=False)
        ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax0.legend(prop=legpara, numpoints=5, markerscale=markersize)
        ax1.set_xlabel("TBJD [days]", fontsize=xls, weight=xlw, labelpad=xlp)
        ax0.set_ylabel("Normalized flux", fontsize=yls, weight=ylw, labelpad=ylp)
        if not publication: ax0.set_title("TIC{}".format(tic),weight='bold')
        else:
            ax1.tick_params('both', length=7, width=1, which='major')
            ax0.tick_params('both', length=7, width=1, which='major')
        ax1.set_ylabel("Residuals", fontsize=yls, weight=ylw)
        for tick in ax0.xaxis.get_major_ticks():
            tick.label1.set_fontsize(xts) 
            tick.label1.set_weight(xtw)
        for tick in ax0.yaxis.get_major_ticks():
            tick.label1.set_fontsize(yts) 
            tick.label1.set_weight(ytw)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(xts) 
            tick.label1.set_weight(xtw)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(yts) 
            tick.label1.set_weight(ytw)
        if savectrl: 
            save2 = save.split(".")
            if not fulltime: save2 = savename("{}_flux_inc{:d}_n{:d}{}.{}".format(save2[0],int(B['inc']),nspots,addon,save2[1]))
            else: save2 = savename("{}_flux_all_inc{:d}_n{:d}{}.{}".format(save2[0],int(B['inc']),nspots,addon,save2[1]))
            plt.savefig(save2)
            if not displayctrl: plt.close(1)
        if displayctrl: plt.show()

        gc.collect()
        plt.figure(1,figsize=(13, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1],sharex=ax0)
        plt.subplots_adjust(hspace=0,top=0.933,bottom=0.082)
        for i in range(len(components.keys())): 
            ax1.plot(tc+data['mintim'],components["Spot {}".format(i+1)]-maxcom[i],"C{}".format(cnums[i%len(cnums)]),marker='.',linestyle='',label="Spot {}".format(i+1),ms=1)
        if fulltime:
            ax0.plot(data['ort']+data['mintim'], data['orf'], "k.", ms=2, label="Data")
            lll = "Model for selected interval"
        else:
            ax0.plot(data['timeorig'], data['fluxorig']/1000+1, "k.", ms=2, label="Data")
            lll = "Model"
        if isgap(t,gv) or (isgap(data['ort'],gv) and fulltime): ax0.plot(t2+data['mintim'], pmx.eval_in_model(flux_model2,map_soln, model=model),marker='.',color="limegreen",label="Missing",ms=1,linestyle='')
        if fulltime: ax0.plot(data['ort']+data['mintim'],flux_model4,'C9.',label="Light curve model",ms=1)
        ax0.plot(data['timeorig'], flux_model_orig, "C1.", label=lll, ms=1,linestyle='')
        if initctrl: ax0.plot(t+data['mintim'], flux_model_guess,marker='.',color="dodgerblue", alpha=0.6, label="Initial model",ms=1,linestyle='')
        plt.setp(ax0.get_xticklabels(), visible=False)
        ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax0.legend(prop=legpara, numpoints=5, markerscale=markersize)
        ax1.legend(prop=legpara, numpoints=5, markerscale=markersize)
        ax1.set_xlabel("TBJD [days]", fontsize=xls, weight=xlw, labelpad=xlp)
        ax0.set_ylabel("Normalized flux", fontsize=yls, weight=ylw, labelpad=ylp)
        if not publication: ax0.set_title("TIC{}".format(tic),weight='bold')
        else:
            ax1.tick_params('both', length=7, width=1, which='major')
            ax0.tick_params('both', length=7, width=1, which='major')
        ax1.set_ylabel("Relative flux", fontsize=yls, weight=ylw)
        for tick in ax0.xaxis.get_major_ticks():
            tick.label1.set_fontsize(xts) 
            tick.label1.set_weight(xtw)
        for tick in ax0.yaxis.get_major_ticks():
            tick.label1.set_fontsize(yts) 
            tick.label1.set_weight(ytw)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(xts) 
            tick.label1.set_weight(xtw)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(yts) 
            tick.label1.set_weight(ytw)
        if savectrl: 
            save2 = save.split(".")
            if not fulltime: save2 = savename("{}_components_inc{:d}_n{:d}{}.{}".format(save2[0],int(B['inc']),nspots,addon,save2[1]))
            else: save2 = savename("{}_components_all_inc{:d}_n{:d}{}.{}".format(save2[0],int(B['inc']),nspots,addon,save2[1]))
            plt.savefig(save2)
            if not displayctrl: plt.close(1)
        if displayctrl: plt.show()

        del flux_model2

        gc.collect()
        plt.figure(1,figsize=(13, 8))
        if fulltime:
            plt.plot(data['ort']+data['mintim'], data['orf'], "k.", ms=2, label="Data")
            plt.plot(data['ort']+data['mintim'],flux_model4,marker=".",color='C3',label="Light curve model",ms=msc,linestyle='')
            tcon = data['ort']+data['mintim']
            if np.max(data['orf']) > 1.02*mf: plt.ylim(top=1.02*mf)
        else:
            plt.plot(data['timeorig'], data['fluxorig']/1000+1, "k.", ms=2, label="Data")
            plt.plot(data['timeorig'], flux_model_orig, marker=".",color='C3', label="Light curve model",ms=msc,linestyle='')
            tcon = data['timeorig']
            if np.max(data['fluxorig']/1000+1) > 1.02*mf: plt.ylim(top=1.02*mf)
        mfcon = np.array([mf for _ in tcon])
        if nspots != 1:
            for i in range(len(components.keys())): plt.plot(tcon,mfcon+(components["Spot {}".format(i+1)]-maxcom[i]),"C{}".format(cnums[i%len(cnums)]),marker = '.',label="Spot {}".format(i+1),ms=msc,linestyle='')
        plt.gca().get_xaxis().set_major_formatter(ScalarFormatter(useOffset=False))
        plt.gca().get_yaxis().set_major_formatter(ScalarFormatter(useOffset=False))
        plt.xlabel("TBJD [days]", fontsize=xls, weight=xlw, labelpad=xlp)
        plt.ylabel("Normalized flux", fontsize=yls, weight=ylw, labelpad=ylp)
        plt.yticks(fontsize=yts,weight=ytw)
        plt.xticks(fontsize=xts,weight=xtw)
        if not publication: plt.title("TIC{}".format(tic),weight='bold')
        else: plt.tick_params('both', length=7, width=1, which='major')
        plt.legend(prop=legpara, numpoints=5, markerscale=markersize)
        if savectrl: 
            save2 = save.split(".")
            if not fulltime: save2 = savename("{}_contribution_inc{:d}_n{:d}{}.{}".format(save2[0],int(B['inc']),nspots,addon,save2[1]))
            else: save2 = savename("{}_contribution_all_inc{:d}_n{:d}{}.{}".format(save2[0],int(B['inc']),nspots,addon,save2[1]))
            plt.savefig(save2)
            if not displayctrl: plt.close(1)
        if displayctrl: plt.show()

        del flux_model4

        gc.collect()
        rec = starry.Map(ydeg=B["ydeg"][0], udeg=B["udeg"][0], inc=B['inc'], amp=ampl, alpha=map_soln['alpha'])
        rec[1] = B["u"][0]
        rec[2] = B["u"][1]
        for i in range(1,nspots+1):
            rec.add_spot(amp=map_soln['amp{:d}'.format(i)]*map_soln['scale'],sigma=map_soln['sigma{:d}'.format(i)],lat=map_soln['lat{:d}'.format(i)],lon=map_soln['lon{:d}'.format(i)])
        if displayctrl:
            rec.show(image=image,projection='rect')
            rec.show(theta=np.abs(t_ani/B['prot'])*360,figsize=(4,4))
        if savectrl: 
            save2 = save.split(".")
            save2 = savename("{}_anim_sphere_grid_inc{:d}_n{:d}{}.{}".format(save2[0],int(B['inc']),nspots,addon,'gif'))
            rec.show(theta=t_ani/B['prot']*360,file=save2,figsize=(4,4))
            save2 = save.split(".")
            sfil = savename("{}_anim_map_inc{:d}_n{:d}{}.gif".format(save2[0],int(B['inc']),nspots,addon))
            save_gif(sfil,image,t_ani,gsp,tic,data['timeorig'][whfl]-data['mintim'],(data['fluxorig'][whfl]/1000+1),flux_model_orig[whfl],data['timeorig'][whfl][0],data['mintim'],np.min(imageshow),np.max(imageshow))
            sfil = savename("{}_anim_aitoff_inc{:d}_n{:d}{}.gif".format(save2[0],int(B['inc']),nspots,addon))
            save_gif(sfil,image,t_ani,gsp,tic,data['timeorig'][whfl]-data['mintim'],(data['fluxorig'][whfl]/1000+1),flux_model_orig[whfl],data['timeorig'][whfl][0],data['mintim'],np.min(imageshow),np.max(imageshow),aitoff=True)

        gc.collect()
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        if not publication:
            if not tbjdctrl: ax.set_title('TIC{}\nTBJD = {:.4f}'.format(tic,t[0]+data['mintim']),weight='bold')
            else: ax.set_title('TIC{}\nTBJD = {}'.format(tic,tbjdnam),weight='bold')  
        else: ax.tick_params('both', length=7, width=1, which='major')         
        ax.set_xlabel('Longitude [deg]',fontsize=xls, weight=xlw, labelpad=xlp)
        ax.set_ylabel('Latitude [deg]', fontsize=yls, weight=ylw, labelpad=ylp)
        ax.imshow(imageshow,origin="lower",extent=(-180, 180, -90, 90),cmap="plasma")
        if publication: plt.subplots_adjust(bottom=0.18,right=1,top=0.967)
        ax.grid(linestyle='--')        
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(xts) 
            tick.label1.set_weight(xtw)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(yts) 
            tick.label1.set_weight(ytw)
        if savectrl: 
            save2 = save.split(".")
            save2 = savename("{}_map_inc{:d}_n{:d}{}.{}".format(save2[0],int(B['inc']),nspots,addon,save2[1]))
            plt.savefig(save2)
            if not displayctrl: plt.close()
        if displayctrl: plt.show()

        gc.collect()
        X,Y = np.meshgrid(np.linspace(-np.pi,np.pi,300),np.linspace(-np.pi/2,np.pi/2,300))
        plt.figure(1,figsize=(12,7))
        plt.subplot(111,projection='aitoff')
        if not publication:
            if not tbjdctrl: plt.title('TIC{}\nTBJD = {:.4f}\n'.format(tic,t[0]+data['mintim']),weight='bold')
            else: plt.title('TIC{}\nTBJD = {}\n'.format(tic,tbjdnam),weight='bold')
        plt.pcolormesh(X,Y[::-1],imageshow[::-1],shading='auto',cmap='plasma')
        plt.grid(True)
        plt.xlabel("\nLongitude [deg]",fontsize=xls,weight=xlw)
        plt.ylabel("Latitude [deg]\n",fontsize=yls,weight=ylw)
        if publication: plt.subplots_adjust(bottom=0.088,right=0.99,left=0.162,top=0.955)
        plt.yticks(fontsize=yts,weight=ytw)
        plt.xticks(fontsize=xts,weight=xtw)
        if savectrl: 
            save2 = save.split(".")
            save2 = savename("{}_aitoff_inc{:d}_n{:d}{}.{}".format(save2[0],int(B['inc']),nspots,addon,save2[1]))
            plt.savefig(save2)
            if not displayctrl: plt.close(1)
        if displayctrl: plt.show()

    if savectrl and not nosphere:
        tsph = np.linspace((data['timeorig'][whfl][0]-data['mintim'])/B['prot']*360,(data['timeorig'][whfl][-1]-data['mintim'])/B['prot']*360,50)
        imagesphere = np.zeros((len(tsph),res,res))
        vmin, vmax = 1000., -1000.
        for n in progress_bar(range(len(tsph)),"Preparing sphere images"): 
            imagesphere[n] = secimshow.render(theta=tsph[n]).eval()
            hel = np.asarray(imagesphere[n],dtype=str)
            hel[np.where(hel == 'nan')] = 0
            if np.max(np.asarray(hel,dtype=float)) > vmax: vmax = np.max(np.asarray(hel,dtype=float))
            hel = np.asarray(imagesphere[n],dtype=str)
            hel[np.where(hel == 'nan')] = 100000.
            if np.min(np.asarray(hel,dtype=float)) < vmin: vmin = np.min(np.asarray(hel,dtype=float))
            gc.collect()
        save2 = save.split(".")
        sfil = savename("{}_anim_sphere_inc{:d}_n{:d}{}.gif".format(save2[0],int(B['inc']),nspots,addon))
        save_gif(sfil,imagesphere,t_ani,gsp*2,tic,data['timeorig'][whfl]-data['mintim'],(data['fluxorig'][whfl]/1000+1),flux_model_orig[whfl],data['timeorig'][whfl][0],data['mintim'],vmin,vmax,sphere=True)
        del imagesphere
        
    del flux_model_orig
    gc.collect()
    if sample:
        if not diffctrl: del map_soln['alpha']
        if not fscale: del map_soln['scale']
        with model:
            if platform != 'darwin': 
                trace = pm.sample(tune=1000,draws=2500,start=map_soln,chains=nchains,cores=ncores,return_inferencedata=False, target_accept=0.9)
            else:
                import multiprocessing as mp
                trace = pm.sample(tune=1000,draws=2500,start=map_soln,chains=1,cores=ncores,mp_ctx=mp.get_context("fork"),return_inferencedata=False, target_accept=0.9)
        if not diffctrl: map_soln['alpha'] = alpha
        if not fscale: map_soln['scale'] = scale

        with model: 
            display(pm.summary(trace, var_names=[*list(sspots.keys()),*scalealphanames]))

        if savectrl or displayctrl:
            for i in range(1,nspots+1):
                truths_ms=[map_soln['amp{:d}'.format(i)],map_soln['sigma{:d}'.format(i)],map_soln['lat{:d}'.format(i)],map_soln['lon{:d}'.format(i)]]
                samples = pm.trace_to_dataframe(trace, varnames=['amp{:d}'.format(i),'sigma{:d}'.format(i),'lat{:d}'.format(i),'lon{:d}'.format(i),*scalealphanames])
                #plt.figure(1,figsize=(12, 8))
                corner(samples,truths=truths_ms)
                if savectrl: 
                    save2 = save.split(".")
                    save2 = savename("{}_corner_inc{:d}_n{:d}_spot{}{}.{}".format(save2[0],int(B['inc']),nspots,i,addon,save2[1]))
                    plt.savefig(save2)
                    if not displayctrl: plt.close(1)
                if displayctrl: plt.show()

            plt.figure(1,figsize=(12, 5))
            plt.plot(t+data['mintim'], flux, "k.", ms=2, label="data")
            label = "Samples"
            for i in np.random.choice(range(len(trace["flux_model"])), 24):
                plt.plot(t+data['mintim'], trace["flux_model"][i], "C0-", alpha=0.3, label=label)
                label = None
            plt.plot(t+data['mintim'],np.mean(trace["flux_model"],axis=0),color='C3',linestyle='--',alpha=0.7,label='Mean model')
            plt.legend(prop=legpara, numpoints=5, markerscale=markersize)
            if not publication: plt.title("TIC{}".format(tic),weight='bold')
            else: plt.tick_params('both', length=7, width=1, which='major')
            plt.xlabel("time [days]",fontsize=xls,weight=xlw,labelpad=xlp)
            plt.ylabel("Normalized flux",fontsize=xls,weight=xlw,labelpad=ylp)
            plt.gca().get_xaxis().set_major_formatter(ScalarFormatter(useOffset=False))
            plt.gca().get_yaxis().set_major_formatter(ScalarFormatter(useOffset=False))
            plt.yticks(fontsize=yts,weight=ytw)
            plt.xticks(fontsize=xts,weight=xtw)
            if savectrl: 
                save2 = save.split(".")
                save2 = savename("{}_sample_inc{:d}_n{:d}{}.{}".format(save2[0],int(B['inc']),nspots,addon,save2[1]))
                plt.savefig(save2)
                if not displayctrl: plt.close(1)
            if displayctrl: plt.show()

            mapmu = starry.Map(ydeg=B["ydeg"][0],udeg=B["udeg"][0],inc=B["inc"], amp=ampl)
            for i in range(1,nspots+1):
                mapmu.add_spot(amp=np.mean(trace['amp{:d}'.format(i)])*scale,sigma=np.mean(trace['sigma{:d}'.format(i)]),lat=np.mean(trace['lat{:d}'.format(i)]),lon=np.mean(trace['lon{:d}'.format(i)]))
            mapmu[1] = B["u"][0]
            mapmu[2] = B["u"][1]
            mu=mapmu.render(projection="rect").eval()
            if filterctrl: mu = zebra_filter(mu)

            _, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].set_title('First model',weight='bold')
            ax[0].set_xlabel('Longitude [deg]',fontsize=xls,weight=xlw)
            ax[0].set_ylabel('Latitude [deg]',fontsize=yls,weight=ylw)
            ax[0].imshow(imageshow,origin="lower",extent=(-180, 180, -90, 90),cmap="plasma")
            ax[1].set_title('Mean model',weight='bold')
            ax[1].set_xlabel('Longitude [deg]',fontsize=xls,weight=xlw)
            ax[1].set_ylabel('Latitude [deg]',fontsize=yls,weight=ylw)
            ax[0].set_title("TIC{}".format(tic),weight='bold')
            ax[1].imshow(mu,origin="lower",extent=(-180, 180, -90, 90),cmap="plasma")
            for tick in ax[0].xaxis.get_major_ticks():
                tick.label1.set_fontsize(xts) 
                tick.label1.set_weight(xtw)
            for tick in ax[0].yaxis.get_major_ticks():
                tick.label1.set_fontsize(yts) 
                tick.label1.set_weight(ytw)
            for tick in ax[1].xaxis.get_major_ticks():
                tick.label1.set_fontsize(xts) 
                tick.label1.set_weight(xtw)
            for tick in ax[1].yaxis.get_major_ticks():
                tick.label1.set_fontsize(yts) 
                tick.label1.set_weight(ytw)
            if savectrl: 
                save2 = save.split(".")
                save2 = savename("{}_mean-map_inc{:d}_n{:d}{}.{}".format(save2[0],int(B['inc']),nspots,addon,save2[1]))
                plt.savefig(save2)
                if not displayctrl: plt.close()
            if displayctrl: plt.show()
    return map_soln

if __name__ == "__main__":
    init()
    if "./" in argv[0]: argv[0] = argv[0][2:]
    elif "/" in argv[0]: argv[0] = argv[0].split("/")[-1]
    params, save = 'star_params.npz', "sspots.pdf"
    command, publication, nth = " ".join(argv), False, 1
    alu, slu = np.array((-0.1,0.0)), np.array((0.0,0.05))
    sample, noadap, nspots, force = False, False, 1, False
    lalu, lolu = np.array((-70.0,70.0)), np.array((-180.0,180.0))
    prec, dprec, temper, temperctrl = 0.000107, 0.00005, 5777, False
    adap, nstep, gsp, falpha, falphactrl, terr = 5, 30, 1, 0, False, 0.0
    modelfile, modelctrl, staractrl, modelonly = "spots.txt", False, False, False
    def_inclination, finc, nosphere, nochi, terrctrl = 50, False, False, False, False
    diffctrl, filterctrl, fulltime, displayctrl, savectrl = False, False, False, True, False
    alpha, prs, ylm, omb, mprod, gv, obm, stara  = [0.,1.], 0.1, 25, 0.015, 8, 0.2, 0.005, 1
    tbjd, tbjdctrl, initctrl, ncores, nchains, pran, liveplot = 0, False, False, 4, 4, False, False
    chiv, logpv, chivctrl, logpvctrl, check, pcom, ps, fscale = 10000, 10000, False, False, False, False, False, False
    for arg in argv:    
        if "-star=" in arg: params = arg.split("=")[-1]
        elif "-nspots=" in arg: nspots = abs(str_to_num(arg.split("=")[-1],int,"nspots"))
        elif "-prec=" in arg: prec = abs(str_to_num(arg.split("=")[-1],float,"prec"))
        elif "-dprec=" in arg: dprec = abs(str_to_num(arg.split("=")[-1],float,"dprec"))
        elif "-pers=" in arg: prs = abs(str_to_num(arg.split("=")[-1],float,"pers"))
        elif "-omb=" in arg: omb = abs(str_to_num(arg.split("=")[-1],float,"omb"))
        elif "-obm=" in arg: obm = abs(str_to_num(arg.split("=")[-1],float,"obm"))
        elif "-finc" in arg:
            finc = True
            if "=" in arg: def_inclination = abs(str_to_num(arg.split("=")[-1],float,"def_inclination"))
        elif "-gv=" in arg: gv = abs(str_to_num(arg.split("=")[-1],float,"gv"))
        elif "-ylm=" in arg:
            ylm = abs(str_to_num(arg.split("=")[-1],int,"ylm"))
            if ylm > 50:
                print("\n\a# Error! Ylm can't be greater than 50.")
                exit()
        elif "-ampl=" in arg: stara, staractrl = abs(str_to_num(arg.split("=")[-1],float,"ampl")), True
        elif "-almin=" in arg: alpha[0] = abs(str_to_num(arg.split("=")[-1],float,"almin"))
        elif "-almax=" in arg: alpha[1] = abs(str_to_num(arg.split("=")[-1],float,"almax"))
        elif "-save" in arg:
            savectrl = True
            if '=' in arg and arg[-1] != '=': save = arg.split("=")[-1]
        elif "-adap=" in arg: adap = abs(str_to_num(arg.split("=")[-1],int,"adap"))
        elif "-nstep=" in arg: nstep = str_to_num(arg.split("=")[-1],int,"nstep")
        elif "-mprod=" in arg: mprod = abs(str_to_num(arg.split("=")[-1],float,"mprod"))
        elif "-amin=" in arg: alu[0] = str_to_num(arg.split("=")[-1],float,"amin")
        elif "-amax=" in arg: alu[1] = str_to_num(arg.split("=")[-1],float,"amax")
        elif "-smin=" in arg: 
            slu[0] = str_to_num(arg.split("=")[-1],float,"smin")
            if slu[0] > 1: slu[0] /= 100
        elif "-smax=" in arg: 
            slu[1] = str_to_num(arg.split("=")[-1],float,"smax")
            if slu[1] > 1: slu[1] /= 100
        elif "-lamin=" in arg:
            lalu[0] = str_to_num(arg.split("=")[-1],float,"lamin")
            if abs(lalu[0]) > 90:
                print("\a\n# Error! Abs(Latitude) can't be greater than 90")
                exit()
        elif "-lamax=" in arg:
            lalu[1] = str_to_num(arg.split("=")[-1],float,"lamax")
            if abs(lalu[1]) > 90:
                print("\a\n# Error! Abs(Latitude) can't be greater than 90")
                exit()
        elif "-lomin=" in arg:
            lolu[0] = str_to_num(arg.split("=")[-1],float,"lomin")
            if abs(lolu[0]) > 180:
                print("\a\n# Error! Abs(Longitude) can't be greater than 180")
                exit()
        elif "-lomax=" in arg:
            lolu[1] = str_to_num(arg.split("=")[-1],float,"lomax")
            if abs(lolu[1]) > 180:
                print("\a\n# Error! Abs(Longitude) can't be greater than 180")
                exit()
        elif "-gsp=" in arg: gsp = str_to_num(arg.split("=")[-1],int,"gsp")
        elif "-model" in arg:
            modelctrl = True
            if "=" in arg and arg[-1] != '=': modelfile = arg.split("=")[1] 
        elif "-falpha=" in arg: falpha, falphactrl = abs(str_to_num(arg.split("=")[-1],float,"falpha")), True
        elif "-tbjd=" in arg: tbjd, tbjdctrl = str_to_num(arg.split("=")[-1],float,"tbjd"), True
        elif '-ncores=' in arg: ncores = abs(str_to_num(arg.split("=")[-1],int,"ncores"))
        elif '-nchains=' in arg: nchains = abs(str_to_num(arg.split("=")[-1],int,"nchains"))
        elif "-chi=" in arg: chiv, chivctrl = abs(str_to_num(arg.split("=")[-1],float,"chi")), True
        elif "-logp=" in arg: logpv, logpvctrl = str_to_num(arg.split("=")[-1],float,"logpv"), True
        elif "-temp=" in arg: temper, temperctrl = abs(str_to_num(arg.split("=")[-1],float,"temperature")), True
        elif "-terr=" in arg: terr, terrctrl = abs(str_to_num(arg.split("=")[-1],float,"temperature error")), True
        elif "-nth=" in arg: nth = abs(str_to_num(arg.split("=")[-1],int,"nth"))
        elif arg == "--pran": pran = True
        elif arg == "--filter": filterctrl = True
        elif arg == "--init": initctrl = True
        elif arg == "--diff": diffctrl= True
        elif arg == "--nodisp": displayctrl = False
        elif arg == "--force": force = True
        elif arg == "--sample": sample = True
        elif arg == "--noadap": noadap = True
        elif arg == '--fulltime': fulltime = True
        elif arg == '--nosphere': nosphere = True
        elif arg == '--nochi': nochi = True
        elif arg == '--check': check = True
        elif arg == '--ps': ps = True
        elif arg == "--fscale": fscale = True
        elif arg == '--pub': publication = True
        elif arg == '--pcom': pcom = True
        elif arg == '--liveplot': liveplot = True
        elif (("-h" in arg and len(arg) == 2) or arg == "--help"): helpf()

    if "--modelonly" in argv: modelonly = True
    if len(argv) == 1 and not exists(params): helpf()
    if not exists(params):
        print("\a#Error! There is no such file as {} in this folder!".format(params))
        exit()
    check_value(alu,"amplitude")
    check_value(slu,"surface")
    check_value(lalu,"latitude")
    check_value(lolu,"longitude")
    check_value(alpha,"alpha")
    gc.collect()
    
    try:
        recreate_sspots(params,nspots,prec,dprec,prs,omb,obm,def_inclination,gv,ylm,stara,alpha,alu,slu,lalu,lolu,adap,mprod,savectrl,save,force,displayctrl,sample,noadap,finc,gsp,nstep,diffctrl,filterctrl,fulltime,nosphere,command,modelfile,modelctrl,staractrl,tbjd,tbjdctrl,initctrl,ncores,nchains,chiv,logpv,chivctrl,logpvctrl,nochi,temper,temperctrl,check,pcom,ps,fscale,pran,falpha,falphactrl,publication,liveplot,terrctrl,terr,modelonly,nth)
    except KeyboardInterrupt:
        if '--singlecache' not in argv: system("rm -rf {}".format(theano_dir))
        print("\n> Program shut down by user.")