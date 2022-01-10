"""
Perform hyperparameter optimization for a given set of parameters and the given training set. The best set of
parameters is then chosen to train a model on the entire trainings set.
"""

from argparse import ArgumentParser
from typing import List, Tuple, Dict, Union
import logging
import os
import json
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import CDPL.Chem as Chem

from src.qphar import Qphar, LOOKUPKEYS
from src.utils import selectMostRigidMolecule
from DA_Michael.makeTrainTestData import loadMolecules


SEARCH_PARAMETERS = {
    'weightType': ['distance', 'nrOfFeatures', None],
    'threshold': [1, 1.5, 2],
}

GENERAL_PARAMETERS = {
    'modelType': 'randomForest',
    'modelKwargs': {
        'n_estimators': 20,
        'max_depth': 3
    }
}


def parseArgs():
    parser = ArgumentParser()
    parser.add_argument('-iTrain', required=True, type=str, help='training set of type sdf')
    parser.add_argument('-activityName', required=True, type=str, help='name of activity property in sdf file')
    parser.add_argument('-nrCvFolds', required=False, type=int, default=5, help='number of cv folds')
    parser.add_argument('-o', required=True, type=str, help='folder to save results to')
    parser.add_argument('-maxDepth', required=False, type=int, default=None, help='max depth of random forest')
    parser.add_argument('-nEstimators', required=False, type=int, default=None, help='nr estimators of random forest')
    return parser.parse_args()


def scoreQpharModelByR2(qpharModel: Qphar, testSet: List[Chem.BasicMolecule], yTrue: np.array) -> float:
    yPred = qpharModel.predict(testSet)
    return r2_score(yTrue.flatten(), yPred.flatten())


def trainQpharModel(trainingSet: List[Chem.BasicMolecule],
                    parameters: Dict[str, Union[str, int, float]],
                    ) -> Tuple[Union[None, Qphar], float]:
    mostRigidMolecule, remainingMolecules = selectMostRigidMolecule(trainingSet)
    activities = np.array([mol.getProperty(LOOKUPKEYS['activity']) for mol in trainingSet])
    qpharModel = None
    bestScore = 0.0
    for i in range(len(remainingMolecules)):
        model = Qphar(template=[mostRigidMolecule, remainingMolecules[i]], **parameters)
        model.fit([remainingMolecules[j] for j in range(len(remainingMolecules)) if j != i], mergeOnce=True)
        score = scoreQpharModelByR2(model, trainingSet, activities)
        if score > bestScore:
            qpharModel = model
            bestScore = score
    return qpharModel, bestScore


def splitCv(nrSamples: int, nrFolds: int) -> Dict[int, Tuple[List[int], List[int]]]:
    indices = np.arange(nrSamples)
    kfold = KFold(nrFolds)
    folds = {i: fold for i, fold in enumerate(kfold.split(indices))}
    return folds


def combineParameters(searchParams: Dict[str, List[Union[str, float, int]]]) -> List[Dict[str, Union[str, int, float]]]:
    keys = sorted(searchParams.keys())
    combinations = product(*(searchParams[k] for k in keys))
    combinedParameters = [{keys[k]: values[k] for k in range(len(keys))} for values in combinations]
    return combinedParameters


def makeCv(parameters: Dict[str, Union[str, int, float]],
           molecules: List[Chem.BasicMolecule],
           folds: Dict[int, Tuple[List[int], List[int]]],
           ) -> Tuple[float, float]:
    scores = []
    activities = np.array([mol.getProperty(LOOKUPKEYS['activity']) for mol in molecules])
    for i, fold in folds.items():
        trainingFold, testFold = fold[0], fold[1]
        model, trainingScore = trainQpharModel([molecules[j] for j in trainingFold], parameters)
        if model is not None:
            testScore = scoreQpharModelByR2(model, [molecules[j] for j in testFold], activities[testFold])
            scores.append(testScore)
        else:
            scores.append(0)
    scores = np.array(scores)
    return np.mean(scores), np.std(scores)


def gridSearch(folds: Dict[int, Tuple[List[int], List[int]]],
               molecules: List[Chem.BasicMolecule],
               parametersToTest: List[Dict[str, Union[str, int, float]]],
               ) -> Dict[int, Dict[str, float]]:
    scores = {}
    for i, params in enumerate(parametersToTest):
        trainingParameters = {k: v for k, v in GENERAL_PARAMETERS.items()}
        for k, v in params.items():
            if k in ['max_depth', 'n_estimators']:
                trainingParameters['modelKwargs'][k] = v
            else:
                trainingParameters[k] = v
        logging.info('Running parameters: {}'.format(json.dumps(trainingParameters)))
        mean, std = makeCv(trainingParameters, molecules, folds)
        scores[i] = {
            'mean': mean,
            'std': std
        }
    return scores


if __name__ == '__main__':
    args = parseArgs()
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    outputFolder = args.o if args.o.endswith('/') else '{}/'.format(args.o)

    if args.maxDepth is not None:
        GENERAL_PARAMETERS['modelKwargs']['max_depth'] = args.maxDepth
        logging.info('Setting max_depth to {}'.format(args.maxDepth))
    if args.nEstimators is not None:
        GENERAL_PARAMETERS['modelKwargs']['n_estimators'] = args.nEstimators
        logging.info('Setting n_estimators to {}'.format(args.nEstimators))

    molecules = loadMolecules(args.iTrain, args.activityName)
    cvFolds = splitCv(len(molecules), args.nrCvFolds)
    parametersToTest = combineParameters(SEARCH_PARAMETERS)
    logging.info('Testing {} parameter combinations'.format(len(parametersToTest)))

    cvPerformance = gridSearch(cvFolds, molecules, parametersToTest[:1])
    cvPerformance = pd.DataFrame.from_dict(cvPerformance, orient='index')
    cvPerformance['parameters'] = [json.dumps(params) for params in parametersToTest[:1]]
    cvPerformance.sort_values('mean', ascending=False, inplace=True)
    bestModelIndex = cvPerformance.index.values[0]
    logging.info('Best cv performance:\nMean R2: {}\nStd R2: {}'.format(cvPerformance.at[bestModelIndex, 'mean'],
                                                                        cvPerformance.at[bestModelIndex, 'std']))
    logging.info('Best parameters: {}'.format(json.dumps(parametersToTest[int(bestModelIndex)])))
    logging.info('Training model with best parameter on entire training set...')
    finalModel, trainingScore = trainQpharModel(molecules, parametersToTest[int(bestModelIndex)])

    if not os.path.isdir(outputFolder):
        os.makedirs(outputFolder)

    qpharFolder = '{}trainedModel/'.format(outputFolder)
    if not os.path.isdir(qpharFolder):
        os.makedirs(qpharFolder)
    if finalModel is not None:
        finalModel.save(qpharFolder)
        logging.info('Saved trained model to: {}'.format(qpharFolder))
    else:
        logging.info('Could not train a final model on the training set.')

    cvPerformance.to_csv('{}gridSearchResults.csv'.format(outputFolder))
    logging.info('Saved grid search results to: {}gridSearchResults.csv'.format(outputFolder))
