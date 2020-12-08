"""
The purpose of this script is just to evaluate the influence of a training split on the models performance.
It is NOT possible to build and select a final model from cross-validation. Please refer to training-validation-test
script for such purposes.
Cross-validation is often used for parameter search.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pharmacophores_4 import DistanceHyperpharmacophore, SequentialHyperpharmacophore, LOOKUPKEYS, assignActivitiesToMolecules
from utils.utils import numFeaturesBaseline, standardPropertiesBaseline, extractActivityFromMolecule, AlignmentError, make_activity_plot, selectMostRigidMolecule
from utils.ML_tools import analyse_regression, aggregateRegressionCrossValidationResults
from utils.Molecule_tools import SDFReader
from shutil import copy
import matplotlib.pyplot as plt


def main(args):
    # set some parameters
    args.modelType = 'randomForest'
    args.modelKwargs = {'n_estimators': 20, 'max_depth': 3}

    # load file with indices for cross-validation
    with open(args.cvFolds) as f:
        cvFolds = json.load(f)

    if len(cvFolds['1']['test']) == 0:
        return  # no test set -> cannot do CV

    # check whether output folder exists -> otherwise create
    if not os.path.isdir(args.o):
        os.makedirs(args.o)

    # save params and input file to output folder -> makes it easier to track results
    params = {key: value for key, value in args.__dict__.items()}
    with open('{o}params.json'.format(o=args.o), 'w') as f:
        json.dump(params, f, indent=2)
    if not args.notCopyInputFile:
        copy(args.i, '{}inputMolecules.sdf'.format(args.o))
        copy(args.cvFolds, '{}cvFolds.json'.format(args.o))

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
        raise ValueError(
            'Value "{}" for parameter "hpType" not recognized. Please choose one of [sequential, distance].'.format(
                args.hpType))

    # perform CV
    results, featuresBaseline, propsBaseline, predictions = {}, {}, {}, {}
    for i, fold in cvFolds.items():
        print('Running fold {} out of {}...'.format(i, len(cvFolds)))
        trainingSet, testSet = [molecules[k] for k in fold['training']], [molecules[k] for k in fold['test']]
        testActivities = [mol.getProperty(LOOKUPKEYS['activity']) for mol in testSet]
        predictions[i] = {'y_true': testActivities}

        # train model
        models = []
        modelPerformance = {}
        preds = []
        if args.alignToMostRigidMolecule:
            template, remainingMolecules = selectMostRigidMolecule(trainingSet)
            for j in range(len(remainingMolecules)):
                try:
                    model = hpModel([template, remainingMolecules[j]], **{k: v for k, v in args if k != 'logPath'})
                except AlignmentError:
                    continue

                model.fit([remainingMolecules[k] for k in range(len(remainingMolecules)) if k != j], mergeOnce=True)
                models.append(model)

                y_pred = model.predict(testSet)
                preds.append(y_pred)
                modelPerformance[j] = analyse_regression(np.array(testActivities), y_pred)

        else:
            for j in range(len(trainingSet) - 1):
                try:
                    model = hpModel(trainingSet[j: j + 2], **{k: v for k, v in params.items() if k != 'logPath'})
                except AlignmentError:  # could not align the two templates -> go on to next template
                    continue
                model.fit([trainingSet[k] for k in range(len(trainingSet)) if k not in [j, j + 1]], mergeOnce=True)
                models.append(model)

                y_pred = model.predict(testSet)
                preds.append(y_pred)
                modelPerformance[j] = analyse_regression(np.array(testActivities), y_pred)

        # evaluate best training model on test set
        print('Evaluating fold...')
        modelPerformance = pd.DataFrame.from_dict(modelPerformance, orient='index')
        metric = getattr(args, 'metric', 'RMSE')
        modelPerformance.sort_values(metric, inplace=True, ascending=False if metric in ['R2'] else True)
        testPerformance = modelPerformance.iloc[0, :]
        results[i] = testPerformance.to_dict()

        # plot predictions
        y_pred = np.array(preds[modelPerformance.index.values[0]])
        predictions[i]['y_pred'] = y_pred
        fig, _ = make_activity_plot(np.array(testActivities), y_pred)
        fig.savefig('{logPath}predictions_{i}.png'.format(logPath=args.o, i=i))
        plt.close(fig)

        if not args.omitBaselines:
            # make baselines for fold
            print('Making baselines...')
            mlModel = model._initMLModel(args.modelType,
                                         args.modelKwargs)  # ensure we use the same ML algorithm for a fair comparison between datasets
            featuresBaseline[i] = numFeaturesBaseline(trainingSet, testSet, LOOKUPKEYS['activity'], model=mlModel)
            propsBaseline[i] = standardPropertiesBaseline(trainingSet, testSet, LOOKUPKEYS['activity'], model=mlModel)

    # aggregate results of cv
    print('Analyzing cross-validation...')
    results = aggregateRegressionCrossValidationResults(results)
    featuresBaseline = aggregateRegressionCrossValidationResults(featuresBaseline)
    propsBaseline = aggregateRegressionCrossValidationResults(propsBaseline)

    merged = pd.DataFrame()
    merged = pd.concat([merged, pd.DataFrame.from_dict(results, orient='columns')])
    merged = pd.concat([merged, pd.DataFrame.from_dict(featuresBaseline, orient='columns')])
    merged = pd.concat([merged, pd.DataFrame.from_dict(propsBaseline, orient='columns')])
    merged.reset_index(drop=True, inplace=True)
    merged['dataType'] = ['Hyperpharmacophore_mean', 'Hyperpharmacophore_std', 'FeaturesBaseline_mean',
                          'FeaturesBaseline_std', 'PropsBaseline_mean', 'PropsBaseline_std']

    for key, value in args.__dict__.items():
        if key == 'modelKwargs':  # treat separately, since it is a dict
            continue
        merged[key] = value

    print('Results: ')
    for metric in ['R2', 'RMSE']:
        print(metric)
        print('Hyperpharmacophore:', results[metric])
        print('Number of pharmacophore features baseline:', featuresBaseline[metric])
        print('Standard properties baseline:', propsBaseline[metric])

    # save and plot results
    merged.to_csv('{logPath}cv_results.csv'.format(logPath=args.o), header=True, index=False)

    y_true = np.concatenate([predictions[i]['y_true'] for i in predictions.keys()], axis=0)
    y_pred = np.concatenate([predictions[i]['y_pred'] for i in predictions.keys()], axis=0)
    fig, _ = make_activity_plot(y_true, y_pred)
    fig.savefig('{logPath}activityPlot.png'.format(logPath=args.o))
    plt.close(fig)

    print('Finished cross-validation!')
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
    parser.add_argument('-omitPredictions', action='store_true', default=False,
                        help='indicates whether to save predictions of the model on the training set')
    parser.add_argument('-omitBaselines', action='store_true', default=False,
                        help='whether to include or omit the baselines (standard properties, number of pharmacophore features)')
    parser.add_argument('-cvFolds', type=str, default=None,
                        help='path to file with cv folds')
    parser.add_argument('-alignToMostRigidMolecule', required=False, action='store_true', default=False,
                        help='Indicates whether to align all molecules to the most rigid molecule the training set')
    args = parser.parse_args()

    main(args)
