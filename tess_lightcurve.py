#!/usr/bin/python3

from tqdm import tqdm
from os import system
from PyAstronomy import pyasl
from sys import argv, platform
from lightkurve.correctors import PLDCorrector

import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt

import warnings
from astropy.units import UnitsWarning
warnings.filterwarnings("ignore", category=UnitsWarning)

def helpf():
    print("\n Usage: tess_lightcurve.py <-tic=str> [-sec=int,int,int,...] [-mark=float] [--save] [--long] [--short] [--fast]") 
    print("                             [--nodisp] [--mask] [--sap] [--jd] [--hjd -ra=f64 -dec=f64]\n")
    exit()

def remove_bad_time(time):
    res = [0]
    for i in range(1,len(time)):
        if time[i] > time[res[-1]]:
            res.append(i)

    return res

def main(tic,sec,secctrl,savectrl,mode,maskctrl,beg,sapctrl,figsave,markt,markctrl,ra,dec,hjdctrl,jdctrl):
    print("> Downloading data for {}{}.".format(beg,tic))
    if secctrl: search_result = lk.search_lightcurve('{}{}'.format(beg,tic), mission='TESS', exptime=mode,sector=sec)
    else: search_result = lk.search_lightcurve('{}{}'.format(beg,tic), mission='TESS', exptime=mode)

    if len(search_result) == 0:
        if not secctrl: print("# No data for {}{} (mode {}).".format(beg,tic,mode))
        else: print("# No data for {}{} for sector {} (mode {})".format(beg,tic,sec,mode))
        exit()
    tic, beg = search_result.target_name[0], 'TIC'
    if secctrl and type(sec) is int: tlcs = search_result[0].download_all()
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

    # argso = np.argsort(time)
    # time, flux_pdc, err, utc = time[argso],flux_pdc[argso], err[argso], utc[argso]
    if not secctrl:
        argso = remove_bad_time(time)
        time, flux_pdc, err, utc = time[argso],flux_pdc[argso], err[argso], utc[argso]

    for i in range(len(utc)): utc[i] = utc[i].replace(" ","T")

    if jdctrl: utc = np.array(jd)
    if hjdctrl: utc = np.array([pyasl.helio_jd(jdelement-2.4e6,ra,dec)+2.4e6 for jdelement in jd])
    if maskctrl:
        if not secctrl: 
            masks_result = lk.search_tesscut('{}{}'.format(beg,tic))
            sectors = []
            for element in str(masks_result).split('\n')[5:]:
                if len(lk.search_lightcurve('{}{}'.format(beg,tic), mission='TESS', exptime=mode, sector=int(element.split()[3]))) > 0:
                    sectors.append(int(element.split()[3]))
        else: 
            masks_result = lk.search_tesscut('{}{}'.format(beg,tic),sector=sec)
            sectors = [sec]
        if "Could not resolve" in str(masks_result): print("\a> No mask found for {}{}!".format(beg,tic))
        else:
            if display:
                for mask in tqdm(sectors,"Preparing masks"): 
                    search_result = lk.search_targetpixelfile('{}{}'.format(beg,tic),sector=mask,mission='TESS',exptime=mode)
                    if len(search_result) != 0:
                        tpf = search_result.download(quality_bitmask='default')
                        pld = PLDCorrector(tpf, aperture_mask=tpf.pipeline_mask)
                        pld.correct()
                        wtit = "Sector {:d}    TBJD = {}".format(mask,tpf.time[0].to_value('btjd'))
                        with plt.style.context(lk.MPLSTYLE):
                            _, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True, num = wtit)
                            pld.tpf.plot(ax=axs[0],show_colorbar=False,aperture_mask=pld.aperture_mask,title="aperture_mask",) # Show light curve aperture mask
                            pld.tpf.plot(ax=axs[1],show_colorbar=False,aperture_mask=pld.pld_aperture_mask,title="pld_aperture_mask",) # Show PLD pixel mask
                            pld.tpf.plot(ax=axs[2],show_colorbar=False,aperture_mask=pld.background_aperture_mask,title="background_aperture_mask",)

    if display:    
        print("> Plotting data.")
        plt.subplots(figsize=(14,7.5))
        plt.plot(time,flux_pdc/1000+1,"C0.",ms=2)
        # plt.plot(time,flux_pdc/1000+1,"C0")
        plt.subplots_adjust(top=0.95,bottom=0.095,left=0.09,right=0.98)
        plt.xlabel("TBJD [days]",fontsize=15,weight='bold')
        plt.ylabel("Normalized flux",fontsize=15,weight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title("{}{}".format(beg,tic),weight='bold')
        plt.minorticks_on()
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
        if secctrl: 
            if type(sec) is int: 
                secsav = sec
                if sec < 10: secsav = "0{}".format(sec)
                savename = "lc_{}_s{}_{}.dat".format(tic,secsav,mode)
            else: 
                savename = "lc_{}_".format(tic)
                for element in sec:
                    savename += "s{}{}".format('0'*(2-len(str(int(element)))),int(element))
                savename += "_{}.dat".format(mode)
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
        elif "-sec=" in arg: 
            if ',' not in arg: sec, secctrl = int(arg.split("=")[1]), True
            else: sec, secctrl = np.asarray(arg.split('=')[1].split(','),dtype=int), True
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