import pandas as pd
import numpy as np
import os
import sys
import json
from Molecule_tools import SDFReader
from ML_tools import analyse_regression
from Pharmacophore_tools import get_pharmacophore
sys.path.append('../Scripts/GithubRepo/')
from pharmacophores_4 import DistanceHyperpharmacophore, LOOKUPKEYS, assignActivitiesToMolecules
from utils.utils import extractActivityFromMolecule, AlignmentError, make_activity_plot, selectMostRigidMolecule, ParamsHoldingClass
import matplotlib.pyplot as plt
import CDPL.Chem as Chem
import CDPL.Pharm as Pharm
import CDPL.Math as Math
from itertools import product
import multiprocessing as mp
from Notifications import send_telegram_message
import time


# define some general parameters
NR_PROCESSES = 8
BASEPATH = '../Data/Evaluation_datasets/Phase_paper/Debnath_2002/'
ACTIVITY_NAME = 'IC50(nM)_exp'
PHASE_ACTIVITY_NAME = 'IC50(nM)_PHASE>'


def main(args):
    """
    Loads everything separete from other processes. Therefore, molecules are loaded multiple times, which is a slight
    waste of processing, but nevertheless faster due to the parallel processing. Also, we can handle memory better this
    way.
    :param args:
    :return:
    """
    # define parameters
    args = ParamsHoldingClass(args)

    print('Running', args.__dict__)

    # load data
    r = SDFReader('{b}Debnath_2002_compounds_conformations.sdf'.format(b=BASEPATH))
    molecules, activities = [], []
    for mol in r:
        a = extractActivityFromMolecule(mol, ACTIVITY_NAME)
        if a is None:
            continue
        assignActivitiesToMolecules([mol], [float(a)])
        molecules.append(mol)
        activities.append(a)

    # split into training and test data
    training, test = [], []
    testActivities, trainingActivities = [], []
    trainingIndices, testIndices = [], []
    for i, mol in enumerate(molecules):
        dataBlock = Chem.getStructureData(mol)
        for p in dataBlock:
            if 'Training_set' in p.header:
                if p.data == 'True':
                    training.append(mol)
                    trainingActivities.append(activities[i])
                    trainingIndices.append(i)
                else:
                    test.append(mol)
                    testActivities.append(activities[i])
                    testIndices.append(i)

    # get template from trainin set
    template, remainingMolecules = selectMostRigidMolecule(training)

    # train models
    tempModels = {}
    modelPerformance = {}
    predictions = {}
    trainingPredictions = {}
    trainingSet = [template]
    trainingSet.extend(remainingMolecules)
    for j in range(len(remainingMolecules)):
        try:
            model = DistanceHyperpharmacophore([template, remainingMolecules[j]],
                                               **{k: v for k, v in args if k != 'logPath'})
        except AlignmentError:
            continue

        model.fit([remainingMolecules[k] for k in range(len(remainingMolecules)) if k != j], mergeOnce=True)
        tempModels[j] = model

        y_pred_training = model.predict(trainingSet)
        y_pred = model.predict(test)
        modelPerformance[j] = analyse_regression(np.array(testActivities), y_pred)
        predictions[j] = y_pred
        trainingPredictions[j] = y_pred_training

    if len(modelPerformance) == 0:  # we could not create a single model for the given parameters
        print('Failed to create a model')
        return

    modelPerformance = pd.DataFrame.from_dict(modelPerformance, orient='index')
    modelPerformance.sort_values('R2', inplace=True, ascending=False)
    model = tempModels[modelPerformance.index.values[0]]
    predictions = predictions[modelPerformance.index.values[0]]
    trainingPredictions = trainingPredictions[modelPerformance.index.values[0]]
    modelPerformance = modelPerformance.iloc[0]

    # save results
    key = args.i
    outputPath = args.outputPath
    modelPerformance.to_csv('{}/results_{}.csv'.format(outputPath, key))
    with open('{}/params_{}.json'.format(outputPath, key), 'w') as f:
        json.dump(args.__dict__, f)

    if not os.path.isdir('{}/model_{}/'.format(outputPath, key)):
        os.makedirs('{}/model_{}/'.format(outputPath, key))
    model.save('{}/model_{}/'.format(outputPath, key))

    preds = pd.DataFrame(predictions, columns=['y_pred'])
    preds['y_true'] = testActivities
    preds['mol_indices'] = testIndices
    preds.to_csv('{}/predictions_{}.csv'.format(outputPath, key))

    fig, _ = make_activity_plot(np.array(testActivities), preds['y_pred'].values, xLabel='pIC50 (Exp)', yLabel='pIC50 (Pred)')
    fig.savefig('{}/predictions_{}.png'.format(outputPath, key))
    plt.close()

    preds_training = pd.DataFrame(trainingPredictions, columns=['y_pred'])
    preds_training['y_true'] = trainingActivities
    preds_training['mol_indices'] = trainingIndices
    preds_training.to_csv('{}/training_predictions_{}.csv'.format(outputPath, key))

    fig, _ = make_activity_plot(np.array(trainingActivities), preds_training['y_pred'].values, xLabel='pIC50 (Exp)',
                                yLabel='pIC50 (Pred)')
    fig.savefig('{}/training_predictions_{}.png'.format(outputPath, key))
    plt.close()

    return modelPerformance


def run_parallel(nr_processes, jobs):
    # set up some overhead
    processes = {}  # keep track of all the processes we started
    scheduledJobs = mp.Queue()
    for job in jobs:
        scheduledJobs.put(job)
    finishedJobs = mp.Queue()  # subprocesses fill this queue with process id once finished

    def run(args, process_name):
        main(*args)
        print('Finished main run for process', process_name)
        finishedJobs.put(process_name)
        print('push process finished', process_name)

    # spawn processes
    for i in range(nr_processes):
        name = time.time()
        p = mp.Process(target=run, args=(scheduledJobs.get(), name), name=str(name))
        processes[name] = p
        p.start()  # immediately starts running the process
        print('Started process', name)
        time.sleep(1)

    # kill and restart processes once done to clean up memory and start fresh
    running_processes = len(processes)
    print('processes:', len(processes))
    while running_processes > 0:
        print('processes:', len(processes))
        name = finishedJobs.get(True)  # blocks until we get a new entry
        print('received process', name)
        p = processes.pop(name)
        print('processes:', len(processes))
        p.terminate()
        print('terminated process', name)

        # check whether there is something left to do -> start over
        if not scheduledJobs.empty():
            name = time.time()
            p = mp.Process(target=run, args=(scheduledJobs.get(), name), name=str(name))
            processes[name] = p
            p.start()
            print('started process : ', name)
            print('processes:', len(processes))

        running_processes = len(processes)
        print('processes:', len(processes))

    # all jobs ran successfully -> clean up
    scheduledJobs.close()
    finishedJobs.close()
    for key in processes.keys():  # should be empty already, but better check
        p = processes.pop(key)
        p.terminate()


if __name__ == '__main__':

    # determine output path
    outputPath = 'compare_against_phase_results'
    i = 1
    while os.path.isdir('{f}_{i}/'.format(f=outputPath, i=str(i))):
        i += 1
    filename = '{f}_{i}/'.format(f=outputPath, i=str(i))
    os.makedirs(filename)

    # define model and search parameters
    generalParams = {
        'fuzzy': True,
        'outputPath': filename
    }

    searchParams = {
        'weightType': ['distance', 'nrOfFeatures', None],
        'modelType': ['randomForest', 'ridge', 'pca_ridge', 'pls', 'pca_lr'],
        'threshold': [1, 1.5, 2],
    }

    modelParams = {
        'ridge': {'fit_intercept': False},
        # 'lasso': {'fit_intercept': False},
        'randomForest': {'n_estimators': 10, 'max_depth': 3},  # test various parameters here too
        'pca_ridge': {},
        # 'pca_lasso': {},
        'pls': {},
        'pca_lr': {}
    }

    rfParams = {
        'n_estimators': [10, 15, 20],
        'max_depth': [2, 3]
    }

    keys = sorted(searchParams.keys())
    combinations = product(*(searchParams[k] for k in keys))
    combinedParameters = [{keys[k]: values[k] for k in range(len(keys))} for values in combinations]  # all parameters

    rfkeys = sorted(rfParams.keys())
    rfcombinations = product(*(rfParams[k] for k in rfkeys))
    rfcombinedParameters = [{rfkeys[k]: values[k] for k in range(len(rfkeys))} for values in rfcombinations]  # all parameters

    jobs = []
    for params in combinedParameters:
        if params['modelType'] == 'randomForest':
            for rfParams in rfcombinedParameters:
                # copy parameters
                allParams = {k: v for k, v in generalParams.items()}
                for k, v in params.items():
                    allParams[k] = v
                allParams['modelKwargs'] = rfParams
                allParams['i'] = len(jobs)

                jobs.append((allParams, ))

        else:
            allParams = {k: v for k, v in generalParams.items()}
            for k, v in params.items():
                allParams[k] = v
            allParams['modelKwargs'] = modelParams.get(params['modelType'], {})
            allParams['i'] = len(jobs)

            jobs.append((allParams, ))

    run_parallel(NR_PROCESSES, jobs)
    print('Finished model selection on PHASE-dataset')
    send_telegram_message('Finished model selection on PHASE-dataset')