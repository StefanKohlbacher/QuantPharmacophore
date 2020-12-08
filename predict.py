import sys
import os
import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pharmacophores_4 import DistanceHyperpharmacophore, SequentialHyperpharmacophore, LOOKUPKEYS, assignActivitiesToMolecules
from utils.utils import numFeaturesBaseline, standardPropertiesBaseline, extractActivityFromMolecule, AlignmentError, make_activity_plot
from utils.ML_tools import analyse_regression, aggregateRegressionCrossValidationResults
from utils.Molecule_tools import SDFReader, mol_to_sdf
import CDPL.Chem as Chem
from shutil import copy


def main(args):
    # check whether output folder exists -> otherwise create
    if not os.path.isdir(args.o):
        os.makedirs(args.o)

    # save params and input file to output folder -> makes it easier to track results
    params = {key: value for key, value in args.__dict__.items()}
    with open('{o}params.json'.format(o=args.o), 'w') as f:
        json.dump(params, f, indent=2)
    if not args.notCopyInputFile:
        copy(args.i, '{}inputMolecules.sdf'.format(args.o))

    # load data set
    print('Loading data set...')
    r = SDFReader(args.i, multiconf=True)
    molecules = [mol for mol in r]
    if args.activityName is not None:  # only the case of test set
        activities = [extractActivityFromMolecule(mol, args.activityName) for mol in molecules]
        assignActivitiesToMolecules(molecules, activities)

    # define and load model
    print('Loading model...')
    if args.hpType == 'sequential':
        hpModel = SequentialHyperpharmacophore()
        hpModel.load(args.modelPath)
    elif args.hpType == 'distance':
        hpModel = DistanceHyperpharmacophore()
        hpModel.load(args.modelPath)
    else:
        raise ValueError(
            'Value "{}" for parameter "hpType" not recognized. Please choose one of [sequential, distance].'.format(
                args.hpType))

    print('Make predictions...')
    y_pred = hpModel.predict(molecules)

    # attach predictions to molecules and save
    for pred, mol in zip(y_pred, molecules):
        sdb = Chem.getStructureData(mol)
        sdb.addEntry(' <predicted_value>', str(pred))
        Chem.setStructureData(mol, sdb)
    mol_to_sdf(molecules, '{}predicted_molecules.sdf'.format(args.o), multiconf=True)
    pd.DataFrame(y_pred, columns=['Predictions']).to_csv('{}predictions.csv'.format(args.o))  # additionally save as csv

    if args.includeEvaluation:
        modelPerformance = analyse_regression(np.array(activities), y_pred)
        with open('{}performance.json'.format(args.o), 'w') as f:
            json.dump(modelPerformance, f, indent=2)
        fig, _ = make_activity_plot(np.array(activities), y_pred)
        fig.savefig('{logPath}predictions.png'.format(logPath=args.o))
        plt.close(fig)

    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', required=True, type=str,
                        help='input file -> sdf file with conformations')
    parser.add_argument('-modelPath', required=True, type=str,
                        help='path to folder containing saved hp model')
    parser.add_argument('-o', required=True, type=str,
                        help='output folder -> folder to predictions to')
    parser.add_argument('-hpType', type=str, default='distance',
                        help='type of hyperpharmacophore which the model was trained for. can be one of [sequential, distance]. default: distance')
    parser.add_argument('-notCopyInputFile', action='store_true', default=False,
                        help='indicates to not copy the input file to the output folder -> saves memory, but requires user to keep track of input himself')
    parser.add_argument('-activityName', default=None, type=str,
                        help='name of activity property in sdf file')
    parser.add_argument('-includeEvaluation', action='store_true', default=False,
                        help='indicates whether activities of molecules are known and metrics should be calculated. If so, then activityName needs to be specified too.')

    args = parser.parse_args()
    main(args)
