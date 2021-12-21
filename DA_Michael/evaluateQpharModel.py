"""
Evaluate a trained qphar model on the test set.
"""
import logging
from argparse import ArgumentParser
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from src.qphar import Qphar, LOOKUPKEYS
from DA_Michael.makeTrainTestData import loadMolecules


def parseArgs():
    parser = ArgumentParser()
    parser.add_argument('-model', required=True, type=str, help='path to qphar model folder')
    parser.add_argument('-i', required=True, type=str, help='molecule test set')
    parser.add_argument('-activityName', required=True, type=str, help='name of activity field in sdf')
    parser.add_argument('-o', required=True, type=str, help='output folder')
    return parser.parse_args()


if __name__ == '__main__':
    args = parseArgs()
    molecules = loadMolecules(args.i, args.activityName)
    activities = np.array([mol.getProperty(LOOKUPKEYS['activity']) for mol in molecules])
    model = Qphar()
    model.load(args.model)

    outputFolder = args.o if args.o.endswith('/') else '{}/'.format(args.o)
    if not os.path.isdir(outputFolder):
        os.makedirs(outputFolder)

    yPred = model.predict(molecules)
    r2 = r2_score(activities, yPred.flatten())
    rmse = mean_squared_error(activities, yPred.flatten(), squared=False)
    logging.info('R2-score: {}'.format(round(r2, 3)))
    logging.info('RMSE-score: {}'.format(round(rmse, 3)))

    pd.DataFrame(np.concatenate([yPred.reshape(-1, 1), activities.reshape(-1, 1)], axis=1),
                 columns=['yPred', 'yTrue']).to_csv('{}predictions.csv'.format(outputFolder))
    pd.DataFrame.from_dict({0: {'r2': r2, 'rmse': rmse}}, orient='index').to_csv('{}scores.csv'.format(outputFolder))
