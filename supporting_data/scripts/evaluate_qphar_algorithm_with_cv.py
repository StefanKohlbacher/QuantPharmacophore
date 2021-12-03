import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
import multiprocessing as mp
from argparse import ArgumentParser
from src.qphar import Qphar, assignActivitiesToMolecules, LOOKUPKEYS
from src.ml_tools import analyse_regression, aggregateRegressionCrossValidationResults
from src.molecule_tools import SDFReader
from src.utils import AlignmentError, extractActivityFromMolecule, selectMostRigidMolecule, make_activity_plot, \
    numFeaturesBaseline, standardPropertiesBaseline, ParamsHoldingClass


params = {
    'modelType': 'randomForest',
    'modelKwargs': {'n_estimators': 10, 'max_depth': 3},
    'fuzzy': True,
    'metric': 'R2',
    'weightType': 'distance',
    'mostRigidTemplate': True,
}


def cv(folds, args):
    print('Running {}'.format(args.name))
    # load molecules and split into folds
    r = SDFReader(args.inputFile)
    molecules = [mol for mol in r]
    activities = [extractActivityFromMolecule(mol, 'pchembl_value') for mol in molecules]
    assignActivitiesToMolecules(molecules, activities)

    # make cv
    results = {}
    featuresBaseline = {}
    propsBaseline = {}
    predictions = {}
    for i, fold in folds.items():
        trainingSet, testSet = [molecules[k] for k in fold['training']], [molecules[k] for k in fold['test']]
        testActivities = [mol.getProperty(LOOKUPKEYS['activity']) for mol in testSet]
        predictions[i] = {'y_true': testActivities}

        # iterate over training data and choose different templates to fit HP on
        models = {}
        modelPerformance = {}
        preds = {}
        if args.mostRigidTemplate:
            template, remainingMolecules = selectMostRigidMolecule(trainingSet)
            for j in range(len(remainingMolecules)):
                try:
                    model = Qphar([template, remainingMolecules[j]], **{k: v for k, v in args if k != 'logPath'})
                except AlignmentError:
                    continue

                model.fit([remainingMolecules[k] for k in range(len(remainingMolecules)) if k != j],
                          mergeOnce=getattr(args, 'mergeOnce', None))
                models[j] = model
                y_pred = model.predict(testSet)
                preds[j] = y_pred
                modelPerformance[j] = analyse_regression(np.array(testActivities), y_pred)

        else:
            for j in range(len(trainingSet)-1):
                try:
                    model = Qphar(trainingSet[j: j + 2], **{k: v for k, v in args if k != 'logPath'})
                except AlignmentError:
                    continue
                model.fit([trainingSet[k] for k in range(len(trainingSet)) if k not in [j, j+1]],
                          mergeOnce=getattr(args, 'mergeOnce', None))

                models[j] = model
                y_pred = model.predict(testSet)
                preds[j] = y_pred
                modelPerformance[j] = analyse_regression(np.array(testActivities), y_pred)

        if len(modelPerformance) == 0:
            continue

        # evaluate best training model on validation set
        modelPerformance = pd.DataFrame.from_dict(modelPerformance, orient='index')
        metric = getattr(args, 'metric', 'RMSE')
        modelPerformance.sort_values(metric, inplace=True, ascending=False if metric in ['R2'] else True)
        testPerformance = modelPerformance.iloc[0, :]
        results[i] = testPerformance.to_dict()

        # plot predictions
        y_pred = np.array(preds[modelPerformance.index.values[0]])
        # predictions[i]['y_pred'] = y_pred
        # if not os.path.isdir(args.logPath):
        #     os.makedirs(args.logPath)
        # fig, _ = make_activity_plot(np.array(testActivities), y_pred)
        # fig.savefig('{logPath}predictions_{i}.png'.format(logPath=args.logPath, i=i))
        # plt.close(fig)

        # save predictions
        y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
        y_pred['y_true'] = testActivities
        y_pred['mol_indices'] = fold['test']
        y_pred.to_csv('{}predictions_{i}.csv'.format(args.outputDir, i=i))

        mlModel = model._initMLModel(args.modelType, args.modelKwargs)  # ensure we use the same ML algorithm for a fair comparison between datasets

        # pharmacophore feature baseline
        featuresBaseline[i], featurePredictions = numFeaturesBaseline(trainingSet, testSet, LOOKUPKEYS['activity'], model=mlModel, returnPredictions=True)
        y_pred = pd.DataFrame(featurePredictions, columns=['y_pred'])
        y_pred['y_true'] = testActivities
        y_pred['mol_indices'] = fold['test']
        y_pred.to_csv('{}predictions_numFeaturesBaseline_{i}.csv'.format(args.outputDir, i=i))

        # physico-chemical properties baselines
        propsBaseline[i], propPredictions = standardPropertiesBaseline(trainingSet, testSet, LOOKUPKEYS['activity'], model=mlModel, returnPredictions=True)
        y_pred = pd.DataFrame(propPredictions, columns=['y_pred'])
        y_pred['y_true'] = testActivities
        y_pred['mol_indices'] = fold['test']
        y_pred.to_csv('{}predictions_propsBaseline_{i}.csv'.format(args.outputDir, i=i))

    if len(results) == 0:
        # create empty df and plots. then return
        merged = pd.DataFrame()

        return merged

    # aggregate results of cv
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

    for key, value in args:
        if key == 'modelKwargs':  # treat separately, since it is a dict
            continue
        merged[key] = value

    # save and plot results
    merged.to_csv('{}cv_results.csv'.format(args.outputDir), header=True, index=False)

    # successfully processed assay -> create file to indicate we are done
    open('{}finished.log'.format(args.outputDir), 'w').close()

    return merged


def parseArgs():
    parser = ArgumentParser()
    parser.add_argument('-nrProcesses', default=10, type=int, required=False)
    parser.add_argument('-cvSplit', type=str, required=True, help='filenmame of file containing molecule indices of the cv folds')
    parser.add_argument('-outputName', required=False, default=None, type=str, help='')
    parser.add_argument('-ignoreVersioning', type=bool, required=False, default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    basePath = '../cross_validation'
    filename = 'conformations.sdf'

    inputArgs = parseArgs()

    # determine output
    i = 1
    outputName = inputArgs.outputName if inputArgs.outputName is not None else inputArgs.cvSplit
    if not inputArgs.ignoreVersioning:
        while os.path.isdir('{}/results/{}_{}'.format(basePath, outputName, i)):
            i += 1
    outputDir = '{}/results/{}_{}'.format(basePath, outputName, str(i))
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    jobs = []

    if inputArgs.nrProcesses > 1:
        pool = mp.Pool(inputArgs.nrProcesses)

    for target in os.listdir('{}/splits/'.format(basePath)):
        for assay in os.listdir('{}/splits/{}/'.format(basePath, target)):
            if not os.path.isfile('{}/splits/{}/{}/{}.json'.format(basePath, target, assay, inputArgs.cvSplit)):
                continue

            tempParams = {
                'inputFile': '{}/targets/{}/{}/conformations.sdf'.format(basePath, target, assay),
                'name': '{target}_{assay}'.format(target=target, assay=assay),
                'outputDir': '{}/{}/{}/'.format(outputDir, target, assay),
            }
            if not os.path.isdir(tempParams['outputDir']):
                os.makedirs(tempParams['outputDir'])

            if os.path.isfile('{}finished.log'.format(tempParams['outputDir'])):
                continue

            for key, value in params.items():
                tempParams[key] = value

            with open('{}/splits/{}/{}/{}.json'.format(basePath, target, assay, inputArgs.cvSplit), 'r') as f:
                cvFolds = json.load(f)

            args = ParamsHoldingClass(tempParams)

            if inputArgs.nrProcesses > 1:
                jobs.append(pool.apply_async(cv, args=(cvFolds, args)))

            else:
                jobs.append((cv, (cvFolds, args)))

    first = True
    if inputArgs.nrProcesses > 1:
        for job in jobs:
            cvResults = job.get()

            if not first:
                cvResults.to_csv('{}/cv_results.csv'.format(outputDir), header=False, index=False, mode='a')
            else:
                first = False
                cvResults.to_csv('{}/cv_results.csv'.format(outputDir), header=True, index=False)

        pool.close()

    else:
        for fn, args in jobs:
            cvResults = fn(*args)

            if not first:
                cvResults.to_csv('{}/cv_results.csv'.format(outputDir), header=False, index=False, mode='a')
            else:
                first = False
                cvResults.to_csv('{}/cv_results.csv'.format(outputDir), header=True, index=False)
