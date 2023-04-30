#!/usr/bin/python3

from tqdm import tqdm
from os import system
from PyAstronomy import pyasl
from sys import argv, platform

import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt

def helpf():
    print("\n Usage: tess_lightcurve.py <-tic=str> [-sec=int] [-mark=float] [--save] [--long] [--short] [--fast]") 
    print("                             [--nodisp] [--mask] [--sap] [--jd] [--hjd -ra=f64 -dec=f64]\n")
    exit()

def main(tic,sec,secctrl,savectrl,mode,maskctrl,beg,sapctrl,figsave,markt,markctrl,ra,dec,hjdctrl,jdctrl):
    print("> Downloading data for {}{}.".format(beg,tic))
    if secctrl: search_result = lk.search_lightcurve('{}{}'.format(beg,tic), mission='TESS', exptime=mode,sector=sec)
    else: search_result = lk.search_lightcurve('{}{}'.format(beg,tic), mission='TESS', exptime=mode)
    if len(search_result) == 0:
        if not secctrl: print("# No data for {}{} (mode {}).".format(beg,tic,mode))
        else: print("# No data for {}{} for sector {} (mode {})".format(beg,tic,sec,mode))
        exit()
    tic, beg = search_result.target_name[0], 'TIC'
    if secctrl: tlcs = search_result[0].download_all()
    else: tlcs = search_result.download_all()
    ft = "flux"
    if sapctrl: ft = "sap_flux"
    print("> Data successfully downloaded!")
    time, flux_pdc, err, origflux, utc, jd = [], [], [], [], [], []
    for tlc in tqdm(tlcs,"Quality flag correction"):
        if tlc.flux_origin == 'pdcsap_flux':
            wh = np.where(tlc['quality'] == 0)
            time = [*time,*tlc['time'][wh].to_value(format='btjd')]
            utc = [*utc,*tlc['time'][wh].utc.iso]
            jd = [*jd,*tlc['time'][wh].to_value(format='jd')]
            helpflux = tlc[ft][wh].to_value()
            helperr = tlc[ft+'_err'][wh].to_value()
            mean_flux = np.mean(helpflux)
            origflux = [*origflux,*helpflux]
            helpflux = 1000*(helpflux-mean_flux)/mean_flux
            helperr = helperr/mean_flux*1000
            flux_pdc = [*flux_pdc,*helpflux]
            err = [*err,*helperr]
    time, flux_pdc, err, utc = np.array(time), np.array(flux_pdc), np.array(err), np.array(utc)
    for i in range(len(utc)): utc[i] = utc[i].replace(" ","T")


    if jdctrl: utc = np.array(jd)
    if hjdctrl: utc = np.array([pyasl.helio_jd(jdelement-2.4e6,ra,dec) for jdelement in jd])
    if maskctrl:
        if not secctrl: masks_result = lk.search_targetpixelfile('{}{}'.format(beg,tic),mission="TESS",exptime=mode)
        else: masks_result = lk.search_targetpixelfile('{}{}'.format(beg,tic),sector=sec,mission="TESS",exptime=mode)
        if len(masks_result) == 0: print("\a> No mask found for {}{}!".format(beg,tic))
        else:
            if display:
                for mask in tqdm(masks_result,"Preparing masks"): 
                    wtit = " ".join(str(mask).split()[22:25])+" TBJD = {}".format(tpf.time[0].to_value('btjd'))
                    fig = plt.figure(wtit,figsize=(6,5))
                    tpf = mask.download(quality_bitmask='default')
                    tpf.plot(ax=fig.gca(),aperture_mask=tpf.pipeline_mask)
                    plt.subplots_adjust(left=0.125,bottom=0.11,right=0.933,top=0.88)
    
    if display:    
        print("> Plotting data.")
        plt.subplots(figsize=(14,7.5))
        plt.plot(time,flux_pdc/1000+1,"C0.",ms=2)
        plt.subplots_adjust(top=0.95,bottom=0.095,left=0.09,right=0.98)
        plt.xlabel("TBJD [days]",fontsize=15,weight='bold')
        plt.ylabel("Normalized flux",fontsize=15,weight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title("{}{}".format(beg,tic),weight='bold')
        plt.tick_params('both', length=7, width=1, which='major')
        plt.tick_params('both', length=4, width=1, which='minor')
        if markctrl:
            ylim=plt.ylim()
            if type(markt) is np.ndarray:
                cmap = plt.cm.get_cmap('rainbow')
                marktn = markt-np.min(markt)
                marktn /= np.max(marktn)
                colors = cmap(marktn)
                plt.vlines(markt,ylim[0],ylim[1],colors=colors,linestyles='--')
            else:
                plt.vlines(markt,ylim[0],ylim[1],colors="C3",linestyles='--')
            plt.ylim(ylim)
        if figsave: plt.savefig("TIC{}_{}_flux.pdf".format(tic,mode))
        plt.show()

    if savectrl:
        secsav = sec
        if sec < 10: secsav = "0{}".format(sec)
        if secctrl: savename = "lc_{}_s{}_{}.dat".format(tic,secsav,mode)
        else: savename = "lc_{}_all_{}.dat".format(tic,mode)
        #np.savetxt(savename, np.transpose([time,flux_pdc,err,np.ones(len(time))*np.mean(origflux)]))
        if hjdctrl or jdctrl: np.savetxt(savename, np.transpose([time,flux_pdc,err,utc]))
        else: np.savetxt(savename, np.transpose([time,flux_pdc,err,utc]),fmt=['%20s','%15s','%15s','%27s'])

    if platform != "win32": system("rm -rf ~/.lightkurve-cache/mastDownload/TESS ~/.lightkurve-cache/mastDownload/HLSP &>/dev/null")

if __name__ == "__main__":
    tic, sec = '266744225', 34
    secctrl, savectrl = False, False
    mode = 'short'
    display,sapctrl = True,False
    maskctrl, figsave = False, False
    tictrl, hdctrl = False, False
    markt, markctrl = 0, False
    ra, dec, hjdctrl, jdctrl = 341.707214, 44.333993, False, False

    if len(argv) == 1: helpf()
    for arg in argv:
        if '-tic=' in arg: tic,tictrl = arg.split('=')[1], True
        elif '-hd=' in arg: tic, hdctrl = arg.split('=')[1], True
        elif "-sec=" in arg: sec, secctrl = int(arg.split("=")[1]), True
        elif "-mark=" in arg: 
            if ',' in arg: markt, markctrl = np.array(sorted(np.asarray(arg.split("=")[1].split(','),dtype=float))), True
            else: markt, markctrl = float(arg.split("=")[1]), True
        elif arg == '--save': savectrl = True
        elif arg == '-h' or arg == '--help': helpf()
        elif arg == '--long': mode = 'long'
        elif arg == '--short': mode = 'short'
        elif arg == '--fast': mode = 'fast'
        elif arg == '--mask': maskctrl= True
        elif arg == '--sap': sapctrl = True
        elif arg == '--nodisp': display = False
        elif arg == '--figsave': figsave = True
        elif arg == '--jd': jdctrl = True
        elif arg == '--hjd': hjdctrl = True
        elif '-ra=' in arg: ra = float(arg.split("=")[1])
        elif '-dec=' in arg: ra = float(arg.split("=")[1])

    
    if jdctrl and hjdctrl: 
        print("You can download data in JD or HJD, not in both. Downloading in JD.")
        hjdctrl = False
    if tictrl and not hdctrl: beg = "TIC"
    elif not tictrl and hdctrl: beg = "HD"
    else: helpf()

    if "TIC" in tic: tic = ''.join(tic.split("TIC"))

    try: main(tic,sec,secctrl,savectrl,mode,maskctrl,beg,sapctrl,figsave,markt,markctrl,ra,dec,hjdctrl,jdctrl)
    except KeyboardInterrupt: print("> Program shut down by user.")