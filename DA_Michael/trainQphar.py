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
from makeTrainTestData import loadMolecules


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
    return parser.parse_args()


def scoreQpharModelByR2(qpharModel: Qphar, testSet: List[Chem.BasicMolecule], yTrue: np.array) -> float:
    yPred = qpharModel.predict(testSet)
    return r2_score(yTrue.flatten(), yPred.flatten())


def trainQpharModel(trainingSet: List[Chem.BasicMolecule],
                    parameters: Dict[str, Union[str, int, float]],
                    ) -> Tuple[Union[None, Qphar], float]:
    mostRigidMolecule = selectMostRigidMolecule(trainingSet)
    activities = np.array([mol.getProperty(LOOKUPKEYS['activity']) for mol in trainingSet])
    qpharModel = None
    bestScore = 0.0
    for i in range(len(trainingSet)-1):
        model = Qphar(template=[mostRigidMolecule, trainingSet[i]], **parameters)
        model.fit([trainingSet[j] for j in range(len(trainingSet)-1) if j != i], mergeOnce=True)
        score = scoreQpharModelByR2(model, trainingSet, activities)
        if score > bestScore:
            qpharModel = model
            bestScore = score
    return qpharModel, bestScore


def splitCv(nrSamples: int, nrFolds: int) -> Dict[int, Dict[str, List[int]]]:
    indices = np.arange(nrSamples)
    kfold = KFold(nrFolds)
    folds = {i: fold for i, fold in enumerate(kfold.split(indices))}
    return folds


def combineParameters(searchParams: Dict[str, List[Union[str, float, int]]]) -> List[Dict[str, Union[str, int, float]]]:
    keys = sorted(searchParams.keys())
    combinations = product(*(searchParams[k] for k in keys))
    combinedParameters = [{keys[k]: values[k] for k in range(len(keys))} for values in combinations]
    return combinedParameters


def makeCv(parameters: [Dict[str, Union[str, int, float]]],
           molecules: List[Chem.BasicMolecule],
           folds: Dict[int, Dict[str, List[int]]],
           ) -> Tuple[float, float]:
    scores = []
    activities = np.array([mol.getProperty(LOOKUPKEYS['activity']) for mol in molecules])
    for i, fold in folds:
        trainingFold, testFold = fold['training'], fold['test']
        model, trainingScore = trainQpharModel([molecules[j] for j in trainingFold], parameters)
        if model is not None:
            testScore = scoreQpharModelByR2(model, [molecules[j] for j in testFold], activities[testFold])
            scores.append(testScore)
        else:
            scores.append(0)
    scores = np.array(scores, axis=0)
    return np.mean(scores), np.std(scores)


def gridSearch(folds: Dict[int, Dict[str, List[int]]],
               molecules: List[Chem.BasicMolecule],
               parametersToTest: List[Dict[str, Union[str, int, float]]],
               ) -> Dict[int, Dict[str, float]]:
    scores = {}
    for i, params in enumerate(parametersToTest):
        logging.info('Testing parameters: {}'.format(json.dumps(params)))
        mean, std = makeCv(params, molecules, folds)
        scores[i] = {
            'mean': mean,
            'std': std
        }
    return scores


if __name__ == '__main__':
    args = parseArgs()
    outputFolder = args.o if args.o.endswith('/') else '{}/'.format(args.o)
    molecules = loadMolecules(args.iTrain, args.activityName)
    cvFolds = splitCv(len(molecules), args.nrCvFolds)
    parametersToTest = combineParameters(SEARCH_PARAMETERS)

    cvPerformance = gridSearch(cvFolds, molecules, parametersToTest)
    cvPerformance = pd.DataFrame.from_dict(cvPerformance, orient='index')
    cvPerformance['parameters'] = [json.dumps(params) for params in parametersToTest]
    cvPerformance.sort_values('mean', ascending=False, inplace=True)

    bestModelIndex = cvPerformance.index.values[0]
    finalModel, trainingScore = trainQpharModel(molecules, parametersToTest[bestModelIndex])

    if not os.path.isdir(outputFolder):
        os.makedirs(outputFolder)

    qpharFolder = '{}trainedModel'.format(outputFolder)
    if not os.path.isdir(qpharFolder):
        os.makedirs(qpharFolder)
    if finalModel is not None:
        finalModel.save(qpharFolder)
    else:
        logging.info('Could not train a final model on the training set.')

    cvPerformance.to_csv('{}gridSearchResults.csv'.format(outputFolder))
