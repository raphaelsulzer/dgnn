import sys
import os
# sys.path.insert(0, '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', ''))
import pandas as pd
import numpy as np
import gspread
import gspread_dataframe as gd


def connect():

    try:
        gc = gspread.service_account('../../data-upload-project-277415-eae77da31aa0.json')
        return gc.open("eth3d_results")
    except Exception as e:
        print("Exception: ", e.__class__, "occurred.")
        return None

def gcUpload(new_data,args):

    print("\n######## UPLOAD DATA OF SCENE {} ########".format(args.scene))

    # gc = gspread.service_account('../../data-upload-project-277415-eae77da31aa0.json')
    # sh = gc.open("eth3d_results")
    sh = None
    while (sh is None):
        sh = connect()
    ws = sh.worksheet(args.scene)

    existing = gd.get_as_dataframe(ws,header=0)
    existing.dropna(inplace=True, how='all', axis=0)
    existing.dropna(inplace=True, how='all', axis=1)

    updated = existing.append(new_data)
    updated["Tolerances:"].astype(np.float32)
    updated["Accuracies:"].astype(np.float32)
    updated["Completenesses:"].astype(np.float32)
    updated["F1-scores:"].astype(np.float32)
    updated["Values:"].astype(np.float32)
    gd.set_with_dataframe(ws, updated)

