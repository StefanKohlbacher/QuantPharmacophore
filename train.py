"""
The purpose of this script is to train a model on a given dataset. Multiple models will be returned, with different
initializations.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pharmacophores_4 import DistanceHyperpharmacophore, SequentialHyperpharmacophore, LOOKUPKEYS, assignActivitiesToMolecules
from utils import numFeaturesBaseline, standardPropertiesBaseline, extractActivityFromMolecule, AlignmentError, make_activity_plot, selectMostRigidMolecule
from ML_tools import analyse_regression, aggregateRegressionCrossValidationResults
from Molecule_tools import SDFReader
from shutil import copy
import matplotlib.pyplot as plt


def main(args):
    # set some parameters
    args.modelType = 'randomForest'
    args.modelKwargs = {'n_estimators': 20, 'max_depth': 3}

    # check whether output folder exists -> otherwise create
    if not os.path.isdir(args.o):
        os.makedirs(args.o)

    # save params and input file to output folder -> makes it easier to track results
    params = {key: value for key, value in args.__dict__.items()}
    with open('{o}params.json'.format(o=args.o), 'w') as f:
        json.dump(params, f, indent=2)
    if not args.notCopyInputFile:
        copy(args.i, '{}inputMolecules.sdf'.format(args.o))

    # load training set
    print('Loading training set...')
    r = SDFReader(args.i, multiconf=True)
    molecules = [mol for mol in r]
    activities = [extractActivityFromMolecule(mol, args.activityName) for mol in molecules]
    assignActivitiesToMolecules(molecules, activities)

    # define model
    if args.hpType == 'sequential':
        hpModel = SequentialHyperpharmacophore
    elif args.hpType == 'distance':
        hpModel = DistanceHyperpharmacophore
    else:
        raise ValueError('Value "{}" for parameter "hpType" not recognized. Please choose one of [sequential, distance].'.format(args.hpType))

    # train model and evaluate
    models = []
    modelPerformance = {}
    predictions = []
    figures = []
    if args.alignToMostRigidMolecule:
        template, remainingMolecules = selectMostRigidMolecule(molecules)
        for j in range(len(remainingMolecules)):
            print('Training model {} out of {}...'.format(j + 1, len(remainingMolecules)))
            try:
                model = hpModel([template, remainingMolecules[j]], **{k: v for k, v in params.items() if k != 'logPath'})
            except AlignmentError:
                continue

            model.fit([remainingMolecules[k] for k in range(len(remainingMolecules)) if k != j], mergeOnce=True)
            models.append(model)

            if args.omitEvaluation and not args.savePredictions:  # just train and don't evaluate on training set
                continue

            y_pred = model.predict(molecules)
            predictions.append(y_pred)
            modelPerformance[j] = analyse_regression(np.array(activities), y_pred)

    else:
        for j in range(len(molecules)-1):
            print('Training model {} out of {}...'.format(j+1, len(molecules)-1))
            try:
                model = hpModel(molecules[j: j+2], **{k: v for k, v in params.items() if k != 'logPath'})
            except AlignmentError:  # could not align the two templates -> go on to next template
                continue
            model.fit([molecules[k] for k in range(len(molecules)) if k not in [j, j+1]], mergeOnce=True)
            models.append(model)

            if args.omitEvaluation and not args.savePredictions:  # just train and don't evaluate on training set
                continue

            y_pred = model.predict(molecules)
            predictions.append(y_pred)
            modelPerformance[j] = analyse_regression(np.array(activities), y_pred)

    # evaluate performance
    if not args.omitEvaluation:
        print('Evaluating models...')
        modelPerformance = pd.DataFrame.from_dict(modelPerformance, orient='index')
        performance_values = modelPerformance[args.metric].values
        sortedArgs = np.argsort(performance_values)  # will sort in ascending order
        if args.metric in ['RMSE']:  # reverse
            sortedArgs = [el for el in reversed(sortedArgs)]

        # sort models and predictions
        models = [models[i] for i in sortedArgs]
        predictions = [predictions[i] for i in sortedArgs]

        if args.bestNrOfModelsToSave > 0:
            models = models[:args.bestNrOfModelsToSave]
            predictions = predictions[:args.bestNrOfModelsToSave]

        for i in range(len(models)):
            fig, _ = make_activity_plot(np.array(activities), predictions[i])
            figures.append(fig)

    # save models and results
    print('Saving...')
    for i, model in enumerate(models):
        path = '{o}Model_{i}/'.format(o=args.o, i=i)
        if not os.path.isdir(path):
            os.mkdir(path)
        model.save(path)
        if not args.omitEvaluation:
            figures[i].savefig('{logPath}activityPlot.png'.format(logPath=path))
            modelPerformance.iloc[i].to_csv('{logPath}trainingPerformance.csv'.format(logPath=path))

            if args.omitPredictions:
                predictionsDF = pd.DataFrame(activities, columns=['y_true'])
                predictionsDF['y_pred'] = predictions[i]
                predictionsDF.to_csv('{logPath}trainingPredictions.csv'.format(logPath=path))

    plt.close('all')
    print('Finished training!')
    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', required=True, type=str,
                        help='input file -> sdf file with conformations')
    parser.add_argument('-o', required=True, type=str,
                        help='output folder -> folder to save model to')
    parser.add_argument('-hpType', type=str, default='distance',
                        help='type of hyperpharmacophore to train. can be one of [sequential, distance]. default: distance')
    parser.add_argument('-metric', type=str, default='R2',
                        help='metric to evaluate performance of hyperpharmacophore. possible values: [R2, RMSE]. default: R2')
    parser.add_argument('-notFuzzy', action='store_true', default=False,
                        help='indicates whether pharmacophores are fuzzy or not')
    parser.add_argument('-threshold', type=float, default=1.5,
                        help='threshold below which features are merged')
    parser.add_argument('-notCopyInputFile', action='store_true', default=False,
                        help='indicates to not copy the input file to the output folder -> saves memory, but requires user to keep track of input himself')
    parser.add_argument('-activityName', default='pchembl_value', type=str,
                        help='name of activity property in sdf file')
    parser.add_argument('-omitEvaluation', action='store_true', default=False,
                        help='indicates whether to omit evaluation of the model on the training set. will evaluate per default')
    parser.add_argument('-bestNrOfModelsToSave', default=0, type=int,
                        help='number of top n models to save. 0 (default) indicates all models are saved in descending order')
    parser.add_argument('-omitPredictions', action='store_true', default=False,
                        help='indicates whether to save predictions of the model on the training set')
    parser.add_argument('-alignToMostRigidMolecule', required=False, action='store_true', default=False,
                        help='Indicates whether to align all molecules to the most rigid molecule the training set')

    args = parser.parse_args()
    main(args)
