"""
Evaluate a trained qphar model on the test set.
"""
import logging
from argparse import ArgumentParser
import os

import numpy as np

from src.qphar import Qphar, LOOKUPKEYS
from DA_Michael.makeTrainTestData import loadMolecules
from DA_Michael.trainQphar import scoreQpharModelByR2


def parseArgs():
    parser = ArgumentParser()
    parser.add_argument('-model', required=True, type=str, help='path to qphar model folder')
    parser.add_argument('-i', required=True, type=str, help='molecule test set')
    parser.add_argument('-activityName', required=True, type=str, help='name of activity field in sdf')
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

    score = scoreQpharModelByR2(model, molecules, activities)
    logging.info('R2-score: {}'.format(round(score, 3)))
