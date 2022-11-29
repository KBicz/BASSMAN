#! /usr/bin/python3

import numpy as np
from sys import argv
from os.path import exists
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def helpf():
    print(" Usage: npz_plot.py <npz_file> [--ti] [npz_file2] [--ti] [npz_file3] [--ti] … [--nl]")
    print("                    [-tic=str -p0=float]")
    exit()
    
def main():
    if '-h' in argv or '--help' in argv: helpf()
    if len(argv) == 2: file = argv[-1]
    elif len(argv) == 1: helpf()
    else: file = argv[1:]

    ticctrl = False
    tic='1'
    p0 = 0
    for arg in argv: 
        if '-tic=' in arg: tic, ticctrl = arg.split("=")[-1], True
        elif '-p0=' in arg: p0 = float(arg.split("=")[-1])
    
    if type(file) is str: file = [file]
    i,j = 0,0
    leg = True
    plt.figure(figsize=(10,6))
    try:
        if file[0].split(".")[-1] != "npz":
            title = "TIC{}".format(file[0].split("_")[1])
        else:
            title = "Selected data points"
    except:
        title = "Light curve"
    time = []
    for f in file:
        try:
            if f == '--ti': print("> Time interval for file {} = {:.4f} days ({:d} points)".format(file[i-1],data['t'][-1]-data['t'][0],len(data['t'])))
            elif f == '--nl': leg = False
            else:
                if ".npz" in f:
                    data = np.load(f, allow_pickle=True)
                    plt.plot(data['t']+data['mintim'],data["flux"],'C{:d}.'.format(j),ms=1,label="Selected data from file {}".format(f))
                    ttt = data['t']+data['mintim']
                    if ticctrl: time = [*time,*ttt]
                else:
                    try: 
                        data = np.genfromtxt(f,dtype="float,float,float,float",names=["t","flux","err","sth"])
                        plt.plot(data['t'],data["flux"]/1000+1,'C{:d}.'.format(j),ms=1,label="Data from file {}".format(f))
                    except: 
                        data = np.load(f, allow_pickle=True)
                        plt.plot(data['t']+data['mintim'],data["flux"],'C{:d}.'.format(j),ms=1,label="Data from file {}".format(f))
                    if ticctrl: time = [*time,*data['t']]
            j += 1
        except:
            if '-tic=' in f or '-p0=' in f: 
                pass
            else:
                if not exists(f): print("# Error! File {} does not exists!".format(f))
        i += 1

    if ticctrl:
        try:
            stars = np.genfromtxt("/usr/local/bin/TESS_stars_params.dat",dtype='U13,U6,U10,U10,U10,U10',names=['tic','vsini','rad','pls','prot','temp'])

            wh = np.where(stars['tic'] == tic)
            pls = stars['pls'][wh][0]
            prot = stars['prot'][wh][0]
            if prot == 'nan': prot = pls
            if prot == 'nan': print("# Warning! Rotation period is NaN!")
            prot = float(prot)
            del stars, wh, pls
        except:
            print("\a# Warning! There is no TIC{} in TESS_stars_params.dat file or file is not in folder /usr/local/bin".format(tic))
        
        tstart = np.min(time)
        i = 1
        isline = False
        flux = plt.ylim()
        while tstart <= np.max(time):
            if not isline:
                plt.plot([tstart,tstart],[min(flux),max(flux)],'k--',label="Moment matching p0 = {} TBJD".format(p0))
                isline = True
            else:
                plt.plot([tstart,tstart],[min(flux),max(flux)],'k--')
            plt.text(tstart+0.3*float(prot),max(flux)/2*1.95,str(i),fontsize=15)
            tstart += float(prot)
            i += 1
        plt.ylim(flux)

    plt.xlabel("TBJD [days]",fontsize=20)
    plt.ylabel("Normalized flux",fontsize=20)
    plt.gca().get_xaxis().set_major_formatter(ScalarFormatter(useOffset=False))
    plt.gca().get_yaxis().set_major_formatter(ScalarFormatter(useOffset=False))
    try: plt.title("TIC{}".format(data['tic']),weight="bold")
    except: plt.title(title,weight="bold")
    if leg: plt.legend(fontsize=10, numpoints=5)
    plt.show()
    return 0

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("> Program shut down by user.")