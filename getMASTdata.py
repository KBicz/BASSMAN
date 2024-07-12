#! /usr/bin/python3

import time
import platform
import numpy as np
from sys import argv
from astroquery.mast import Catalogs
from requests.exceptions import HTTPError


if platform.system() == 'Windows':
    import os
    os.system('color')


def helpf():
    print(" Usage: getMASTdata.py <Star name/ TIC number> [-col=int]")
    exit()


def main(star, coln = 0, colctrl = False, nobold = False):
    try: 
        int(star)
        star = "TIC"+star
    except: pass

    keys = ["Teff", "e_Teff", "logg", "e_logg", "rad", "e_rad", "mass", "e_mass", "rho", "e_rho", "lumclass", "lum",
            "e_lum", "d", "e_d"]
    if not nobold:
        start, end = "\033[1m", "\033[0;0m"
    else:
        start, end = '', ''

    catalogTIC = Catalogs.query_object(star, catalog="TIC", radius=0.0001)
    if len(catalogTIC) == 0: 
        print("> Object {} not in TESS inpuit catalogue!".format(star))
        exit()
    elif len(catalogTIC) > 1:

        if not colctrl:
            print("\n\a% Warning! More than one object in the range of search.")
            for x in range(3):
                print(f"Downloading informations about every object in {3-x} s.", end='\r')
                time.sleep(1)
            print()

            maxlen = np.array([len(l.columns) for l in catalogTIC])
            maxlen, maxlenloc = np.max(maxlen), maxlen.argmax()

            for key in catalogTIC[maxlenloc].columns:
                printstr = f"{key}{" "*(13-len(key))}"
                for i in range(len(catalogTIC)):
                    if key in catalogTIC[i].keys() and key not in keys:
                        printstr += " "*(19-len(str(catalogTIC[i][key]))+5) + f"{catalogTIC[i][key]}"
                    elif key in catalogTIC[i].keys() and key in keys:
                        printstr += " "*(19-len(str(catalogTIC[i][key]))+5) + start + f"{catalogTIC[i][key]}" + end
                print(printstr)
        else:
            print()
            catalogTIC = catalogTIC[coln]
            for key in catalogTIC.columns:
                if key in keys:
                    print(start + "{}{}{}".format(key, " " * (18 - len(key)), catalogTIC[key]) + end)
                else:
                    print("{}{}{}".format(key, " " * (18 - len(key)), catalogTIC[key]))
        print()
    else:
        print()
        for key in catalogTIC.columns:
            if key in keys:
                print(start + "{}{}{}".format(key, " "*(18-len(key)), catalogTIC[key].value[0]) + end)
            else:
                print("{}{}{}".format(key, " "*(18-len(key)), catalogTIC[key].value[0]))
        print()

    return 0


if __name__ == "__main__":
    col, nobold, colctrl = 0, '', False
    if (len(argv) == 1 or len(argv) > 4) or '--help' in argv or '-h' in argv:
        helpf()
    else:
        for arg in argv[2:]:
            if arg == '-col=':
                col, colctrl = int(arg.split('=')[1]), True
            elif arg == '--nobold':
                nobold = True

        notdonectrl = True
        while notdonectrl:
            try:
                main(argv[1], col, colctrl, nobold)
                notdonectrl = False
            except HTTPError:
                continue
