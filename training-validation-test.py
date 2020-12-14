import sys
import os
import json
import numpy as np
import pandas as pd
from pharmacophores_4 import DistanceHyperpharmacophore, LOOKUPKEYS, assignActivitiesToMolecules
from utils.utils import numFeaturesBaseline, standardPropertiesBaseline, extractActivityFromMolecule, AlignmentError, make_activity_plot, selectMostRigidMolecule
from utils.ML_tools import analyse_regression, aggregateRegressionCrossValidationResults
from utils.Molecule_tools import SDFReader
from shutil import copy
import matplotlib.pyplot as plt
import multiprocessing as mp


def main(args, trainValidationTestSplit, searchParameters, modelParams):
    # print(args.name)
    print(args['name'])

    outputFolder = args.logPath[:-1] if args.logPath[-1] == '/' else args.logPath
    # combine to be tested parameters
    keys = sorted(searchParameters.keys())
    combinations = product(*(searchParameters[k] for k in keys))
    combinedParameters = [{keys[k]: values[k] for k in range(len(keys))} for values in combinations]  # all parameters

    # load molecules and split into different datasets
    r = SDFReader('{basePath}{d}{f}'.format(basePath=args.basePath, d=args.dataset, f=args.inputFile), multiconf=True)
    molecules = [mol for mol in r]
    activities = [extractActivityFromMolecule(mol, args.activityName) for mol in molecules]
    assignActivitiesToMolecules(molecules, activities)

    # split molecules in training, validation, test set
    trainingSet = [molecules[k] for k in trainValidationTestSplit['training']]
    validationSet = [molecules[k] for k in trainValidationTestSplit['validation']]
    testSet = [molecules[k] for k in trainValidationTestSplit['test']]
    validationActivities = np.array([mol.getProperty(LOOKUPKEYS['activity']) for mol in validationSet])
    testActivities = np.array([mol.getProperty(LOOKUPKEYS['activity']) for mol in testSet])
    template, remainingMolecules = selectMostRigidMolecule(trainingSet)

    # test all parameters on the datasets
    models = {}
    for i, params in enumerate(combinedParameters):
        # print('Running parameters', i, params)
        # set current parameters
        for key, value in params.items():
            args[key] = value
            # setattr(args, key, value)
        args['modelKwargs'] = modelParams[args['modelType']]
        # args.modelKwargs = modelParams[args.modelType]
        args = ParamsHoldingClass(args)

        # create output folder for current parameters
        args.o = '{o}/{i}/'.format(o=outputFolder, i=i)
        # print('Output path', args.o)
        if not os.path.isdir(args.o):
            os.makedirs(args.o)

        # save current parameters
        with open('{o}params.json'.format(o=args.o), 'w') as f:
            json.dump({key: value for key, value in args.__dict__.items()}, f, indent=2)

        # train models on training set
        tempModels = {}
        modelPerformance = {}
        for j in range(len(remainingMolecules)):
            try:
                model = DistanceHyperpharmacophore([template, remainingMolecules[j]], **{k: v for k, v in args if k != 'logPath'})
            except AlignmentError:
                continue
            model.fit([remainingMolecules[k] for k in range(len(remainingMolecules)) if k != j], mergeOnce=True)
            tempModels[j] = model

            y_pred = model.predict(validationSet)  # predict validation set
            modelPerformance[j] = analyse_regression(validationActivities, y_pred)

        # we could not create a single model for the specified train-val-test split with the given params
        if len(modelPerformance) == 0:
            continue

        # select best model from validation set results
        # print('Evaluating model on validation set', args.o)
        modelPerformance = pd.DataFrame.from_dict(modelPerformance, orient='index')
        metric = getattr(args, 'metric', 'R2')
        modelPerformance.sort_values(metric, inplace=True, ascending=False if metric in ['R2'] else True)
        bestModel = tempModels[modelPerformance.index.values[0]]

        # save best model
        models[i] = bestModel

    # test selected models from each parameter setting on test set
    for i, model in models.items():
        o = '{o}/{i}/'.format(o=outputFolder, i=i)

        y_pred = model.predict(testSet)
        modelPerformance = analyse_regression(testActivities, y_pred)
        # evaluate predictions, make plots and save models
        fig, _ = make_activity_plot(testActivities, y_pred)
        fig.savefig('{logPath}predictions.png'.format(logPath=o))
        plt.close()
        pd.DataFrame.from_dict(modelPerformance, orient='index').to_csv('{logPath}performance.csv'.format(logPath=o))
        pd.DataFrame(y_pred, columns=['predictions']).to_csv('{logPath}predictions.csv'.format(logPath=o))
        model.save('{logPath}model/'.format(logPath=o))

    # make baseline models
    mlModel = model._initMLModel(args.modelType, args.modelKwargs)  # use ml-model with same parameters
    perf, pred = numFeaturesBaseline(trainingSet, testSet, LOOKUPKEYS['activity'], model=mlModel, returnPredictions=True)
    pd.DataFrame.from_dict(perf, orient='index').to_csv('{logPath}/performance_features_baseline.csv'.format(logPath=outputFolder))
    pd.DataFrame(pred, columns=['predictions']).to_csv('{logPath}/predictions_features_baseline.csv'.format(logPath=outputFolder))
    perf, pred = standardPropertiesBaseline(trainingSet, testSet, LOOKUPKEYS['activity'], model=mlModel, returnPredictions=True)
    pd.DataFrame.from_dict(perf, orient='index').to_csv('{logPath}/performance_properties_baseline.csv'.format(logPath=outputFolder))
    pd.DataFrame(pred, columns=['predictions']).to_csv('{logPath}/predictions_properties_baseline.csv'.format(logPath=outputFolder))
    print('Finished', args.name)


#####################################################################
# Example script of how to apply main()
#####################################################################


if __name__ == '__main__':
    from utils.utils import ParamsHoldingClass
    from itertools import product

    # define parameterss
    nrProcesses = 8
    basePath = '../../Data/Evaluation_datasets/'
    params = {
        'basePath': basePath,
        'inputFile': 'conformations.sdf',
        'o': 'testOutput',
        'activityName': 'pchembl_value',
        'fuzzy': True,
        # 'threshold': 1.5,  # 1.5 is radius of pharmacophores
        # 'mostRigidTemplate': True,  # True per default
    }
    searchParams = {   # only parameters to be combined!
        'weightType': ['distance', 'nrOfFeatures'],
        'modelType': ['randomForest', 'lasso', 'ridge'],
        'threshold': [1, 1.5, 2]
    }
    modelParams = {
        'ridge': {'fit_intercept': False},
        'lasso': {'fit_intercept': False},
        'randomForest': {'n_estimators': 10, 'max_depth': 3},
        'pca_ridge': {},
        'pca_lasso': {},
        'pls': {}
    }

    # get targets
    # with open('{basePath}evaluation_targets.json'.format(basePath=basePath), 'r') as f:
    #     targetDict = json.load(f)
    targetDict = {
        'GABA_A_a1_b3_g2': {'assayID': ['CHEMBL676826', 'CHEMBL678329', 'CHEMBL822162']},
        'GABA_A_a2_b3_g2': {'assayID': ['CHEMBL1273616', 'CHEMBL681841']},
        # 'GABA_A_a1_b3_g2': {'assayID': ['CHEMBL678329']},
    }
    nrAssaysTotal = 1
    for target, information in targetDict.items():
        nrAssaysTotal *= len(information['assayID'])

    # versioning
    i = 1
    while os.path.isdir('{f}_{i}/'.format(f=params['o'], i=str(i))):
        i += 1
    filename = '{f}_{i}/'.format(f=params['o'], i=str(i))
    os.mkdir(filename)

    jobs = []
    if nrProcesses > 1:
        pool = mp.Pool(min(nrAssaysTotal, nrProcesses))

    # iterate over targets
    for name, information in targetDict.items():
        if not isinstance(information['assayID'], list):
            information['assayID'] = [information['assayID']]

        # iterare over assays
        for assay in information['assayID']:
            tempParams = {
                'dataset': '{target}/{assay}/'.format(target=name, assay=assay),
                'name': '{target}_{assay}'.format(target=name, assay=assay),
                'logPath': '{path}{folder}/{assay}/'.format(path=filename, folder=name, assay=assay),
            }

            # check whether has already been processed --> just in case we have to rerun due to some error, so we do not unnecessary waste resources
            # if os.path.isfile('{logPath}finished.log'.format(logPath=tempParams['logPath'])):  # already processed successfully
            #     continue

            for key, value in params.items():
                tempParams[key] = value

            # load data split
            with open('{basePath}{folder}{cvFolds}.json'.format(basePath=basePath,
                                                                folder=tempParams['dataset'],
                                                                cvFolds='trainValidationTestSplit'), 'r') as f:
                trainValTestSplit = json.load(f)

            # apparently we do not have any test samples -> skip target
            if trainValTestSplit['test'] == 0 or trainValTestSplit['training'] == 0 or trainValTestSplit['validation'] == 0:
                continue

            # args = ParamsHoldingClass(tempParams)

            if nrProcesses > 1:
                jobs.append(pool.apply_async(main, args=(tempParams, trainValTestSplit, searchParams, modelParams)))

            else:
                jobs.append((main, (tempParams, trainValTestSplit, searchParams, modelParams)))

    # make calculations
    if nrProcesses > 1:
        for job in jobs:
            job.get()

        pool.close()

    else:
        for fn, args in jobs:
            fn(*args)
