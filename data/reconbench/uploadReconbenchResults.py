import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', '')) # where the .json file is located
import pandas as pd
import numpy as np
import gspread
import gspread_dataframe as gd





def upload(new_data,args):

    print("\n######## UPLOAD DATA OF SCENE {} ########".format(args.scene))


    gc = gspread.service_account('../../data-upload-project-277415-eae77da31aa0.json')
    sh = gc.open("reconbench_results")  # = "reconbench_results.csv"
    # sh = gc.open("classification_results2")
    ws = sh.worksheet(args.scene)

    existing = gd.get_as_dataframe(ws,header=0)
    existing.dropna(inplace=True, how='all', axis=0)
    existing.dropna(inplace=True, how='all', axis=1)

    updated = existing.append(new_data)
    for name, data in updated.iteritems():
        try:
            updated[name].astype(np.float32)
        except:
            pass

    gd.set_with_dataframe(ws, updated)

