import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
import multiprocessing as mp
from pharmacophores_4 import DistanceHyperpharmacophore, assignActivitiesToMolecules, LOOKUPKEYS
from utilities.ML_tools import analyse_regression, aggregateRegressionCrossValidationResults
from utilities.Molecule_tools import SDFReader
from utilities.utils import AlignmentError, extractActivityFromMolecule, selectMostRigidMolecule, make_activity_plot, \
    numFeaturesBaseline, standardPropertiesBaseline, ParamsHoldingClass

BASEPATH = '/data/local/skohlbacher/GRAIL_QSAR/Data/chembl_targets/'
FILENAME = 'conformations.sdf'
np.random.seed(991)
NR_PROCESSES = min(10, mp.cpu_count())

params = {
    'modelType': 'randomForest',
    'modelKwargs': {'n_estimators': 10, 'max_depth': 3},
    'fuzzy': True,
    'metric': 'R2',
    'weightType': 'distance',
    'mostRigidTemplate': True,
}


def cv(folds, args, hpModel, basePath):
    print('Running {}'.format(args.name))
    # load molecules and split into folds
    r = SDFReader('{basePath}{d}{f}'.format(basePath=basePath, d=args.dataset, f=FILENAME))
    molecules = [mol for mol in r]
    activities = [extractActivityFromMolecule(mol, 'pchembl_value') for mol in molecules]
    assignActivitiesToMolecules(molecules, activities)

    # make cv
    results = {}
    featuresBaseline = {}
    propsBaseline = {}
    predictions = {}
    for i, fold in folds.items():
        trainingSet, testSet = [molecules[k] for k in fold['training']], [molecules[k] for k in fold['_test']]
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
                    model = hpModel([template, remainingMolecules[j]], **{k: v for k, v in args if k != 'logPath'})
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
                    model = hpModel(trainingSet[j: j+2], **{k: v for k, v in args if k != 'logPath'})
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

        # evaluate best training model on _test set
        modelPerformance = pd.DataFrame.from_dict(modelPerformance, orient='index')
        metric = getattr(args, 'metric', 'RMSE')
        modelPerformance.sort_values(metric, inplace=True, ascending=False if metric in ['R2'] else True)
        testPerformance = modelPerformance.iloc[0, :]
        results[i] = testPerformance.to_dict()

        # plot predictions
        y_pred = np.array(preds[modelPerformance.index.values[0]])
        predictions[i]['y_pred'] = y_pred
        if not os.path.isdir(args.logPath):
            os.makedirs(args.logPath)
        fig, _ = make_activity_plot(np.array(testActivities), y_pred)
        fig.savefig('{logPath}predictions_{i}.png'.format(logPath=args.logPath, i=i))
        plt.close(fig)

        # make baselines for fold
        mlModel = model._initMLModel(args.modelType, args.modelKwargs)  # ensure we use the same ML algorithm for a fair comparison between datasets
        featuresBaseline[i] = numFeaturesBaseline(trainingSet, testSet, LOOKUPKEYS['activity'], model=mlModel)
        propsBaseline[i] = standardPropertiesBaseline(trainingSet, testSet, LOOKUPKEYS['activity'], model=mlModel)

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
    merged.to_csv('{logPath}cv_results.csv'.format(logPath=args.logPath), header=True, index=False)

    # successfully processed assay -> create file to indicate we are done
    open('{logPath}finished.log'.format(logPath=args.logPath), 'w').close()

    return merged


def main(params,
         filename,
         hpModel,
         nrProcesses=None,
         targetDict=None,
         basePath=None,
         cvSplit=None,
         ignoreVersioning=False,
         **kwargs):
    if nrProcesses is None:
        nrProcesses = NR_PROCESSES

    if basePath is None:
        basePath = BASEPATH

    # determine filename
    i = 1
    if not ignoreVersioning:
        while os.path.isdir('{f}_{i}/'.format(f=filename, i=str(i))):
            i += 1
    filename = '{f}_{i}/'.format(f=filename, i=str(i))
    os.mkdir(filename)

    # load basic information about datasets
    if targetDict is None:
        with open('{basePath}evaluation_targets.json'.format(basePath=basePath), 'r') as f:
            targetDict = json.load(f)

    jobs = []
    if nrProcesses > 1:
        pool = mp.Pool(min(len(targetDict), nrProcesses))

    with open('{f}params.json'.format(f=filename), 'w') as paramsFile:
        tempParams = {k: v for k, v in params.items()}
        for k, v in kwargs.items():
            tempParams[k] = v
        json.dump(tempParams, paramsFile, indent=2)

    for name, information in targetDict.items():
        if not isinstance(information['assayID'], list):
            information['assayID'] = [information['assayID']]

        for assay in information['assayID']:
            tempParams = {
                'dataset': '{target}/{assay}/'.format(target=name, assay=assay),
                'name': '{target}_{assay}'.format(target=name, assay=assay),
                'logPath': '{path}{folder}/{assay}/'.format(path=filename, folder=name, assay=assay),
            }

            if os.path.isfile('{logPath}finished.log'.format(logPath=tempParams['logPath'])):  # already processed successfully
                continue

            for key, value in params.items():
                tempParams[key] = value
            for key, value in kwargs.items():
                tempParams[key] = value

            if not os.path.isfile('{basePath}{folder}{cvFolds}.json'.format(basePath=basePath, folder=tempParams['dataset'], cvFolds=cvSplit if cvSplit is not None else 'cvFolds')):
                continue
            with open('{basePath}{folder}{cvFolds}.json'.format(basePath=basePath, folder=tempParams['dataset'], cvFolds=cvSplit if cvSplit is not None else 'cvFolds'), 'r') as f:
                cvFolds = json.load(f)

            if cvFolds['1']['_test'] == 0:  # apparently we do not have any _test samples -> skip target
                continue

            args = ParamsHoldingClass(tempParams)

            if nrProcesses > 1:
                jobs.append(pool.apply_async(cv, args=(cvFolds, args, hpModel, basePath)))

            else:
                jobs.append((cv, (cvFolds, args, hpModel, basePath)))

    first = True
    if nrProcesses > 1:
        for job in jobs:
            cvResults = job.get()

            if not first:
                cvResults.to_csv('{f}cv_results.csv'.format(f=filename), header=False, index=False, mode='a')
            else:
                first = False
                cvResults.to_csv('{f}cv_results.csv'.format(f=filename), header=True, index=False)

        pool.close()

    else:
        for fn, args in jobs:
            cvResults = fn(*args)

            if not first:
                cvResults.to_csv('{f}cv_results.csv'.format(f=filename), header=False, index=False, mode='a')
            else:
                first = False
                cvResults.to_csv('{f}cv_results.csv'.format(f=filename), header=True, index=False)


if __name__ == '__main__':

    with open('{basePath}targetDict.json'.format(basePath=BASEPATH), 'r') as f:
        targetDict = json.load(f)

    main(params,
         'training/cv_20-80',
         DistanceHyperpharmacophore,
         nrProcesses=NR_PROCESSES,
         basePath=BASEPATH,
         targetDict=targetDict,
         cvSplit='cvSplit_20-80'
         )
