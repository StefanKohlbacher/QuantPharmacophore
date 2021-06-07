# TODO: script to split datasets into training and test data sets.
#  -> evaluate model and parameter selection with CV on training set, then check performance on test set.
#  Keep the established 80-20 vs 20-80 split, but just add test set

# TODO: also run the script and regenerate the tar files

import os
import json
from argparse import ArgumentParser

import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.molecule_tools import SDFReader
from src.utils import extractActivityFromMolecule


MIN_SAMPLES_TRAINING_SET = 10
MIN_SAMPLES_VALIDATION_SET = 5


def parseArgs():
    parser = ArgumentParser()
    parser.add_argument('-validationSize', required=False, default=0.2, type=float)
    # parser.add_argument('-testSize', required=False, default=0.1, type=float)
    parser.add_argument('-activityName', required=False, default='pchembl_value', type=str)
    parser.add_argument('-name', required=False, default=None, type=str)
    # parser.add_argument('-nrBins', required=False, default=5, type=int)
    parser.add_argument('-nrFolds', required=False, default=5, type=int)
    args = parser.parse_args()
    return args


# def stratifiedTrainTestSplitByActivity(samples, activities, nrTestSamples, nrBins, bins=None):
#     """
#     Put the continuous activity data into bins, which is then used for stratified splitting.
#     """
#     if not bins:
#         # define bins
#         bins = {i: [] for i in range(nrBins)}
#         minValue, maxValue = min(activities), max(activities)
#         binSpan = (maxValue-minValue)/nrBins
#         cutoffs = []
#         for i in range(1, nrBins+1):
#             cutoffs.append(minValue+i*binSpan)
#
#         # populate bins
#         for a, s in zip(activities, samples):
#             for i, c in enumerate(cutoffs):
#                 if a > c:  # if never true -> last bin
#                     break
#             bins[i].append(s)
#
#     # randomly sample test set from bins
#     testSet = []
#     while True:
#         randomBinOrder = np.array(list(bins.keys()))
#         np.random.shuffle(randomBinOrder)
#         for i in randomBinOrder:
#             bin = bins[i]
#             if len(bin) == 0:
#                 bins.pop(i)
#                 continue
#             randomIndex = np.random.choice(np.arange(len(bin)))
#             testSet.append(bin[randomIndex])
#             del bin[randomIndex]
#
#             if len(testSet) == nrTestSamples or len(bins) == 0:
#                 break
#
#         if len(testSet) == nrTestSamples or len(bins) == 0:
#             break
#
#     # put remaining samples into training set
#     trainingSet = [s for s in samples if s not in testSet]
#     np.random.shuffle(trainingSet)
#     np.random.shuffle(testSet)
#     return trainingSet, testSet, bins


def binActivityData(activities, nrBins):
    minValue, maxValue = min(activities), max(activities)
    binSpan = (maxValue - minValue) / nrBins
    cutoffs = []
    for i in range(1, nrBins + 1):
        cutoffs.append(minValue + i * binSpan)

    # create array of bins
    binnedActivityData = []
    for a in activities:
        for binNr, c in enumerate(cutoffs):
            if a <= c:  # if never true -> last bin
                break
        binnedActivityData.append(binNr)

    return binnedActivityData


def stratifiedCVSplitByActivity(activities, nrFolds):
    classes = binActivityData(activities, nrFolds)
    stratifiedSplit = StratifiedKFold(nrFolds, shuffle=True)
    folds = {}
    for train, test in stratifiedSplit.split(activities, classes):
        folds[len(folds)] = {'training': train.tolist(), 'validation': test.tolist()}

    return folds


# def splitDataByActivity(activities, nrFolds):
#     """
#     Split data into training and test set. Then split training set into n-fold CV datasets.
#     """
#     trainingIndices, testIndices, _ = stratifiedTrainTestSplitByActivity(indizes,
#                                                                          activities,
#                                                                          nrTestSamples,
#                                                                          nrBins,
#                                                                          )
#     trainingActivities = [activities[i] for i in trainingIndices]
#
#     folds = stratifiedCVSplitByActivity(trainingActivities, nrFolds, nrBins)
#
#     return folds, testIndices


if __name__ == '__main__':
    basePath = '../cross_validation'
    args = parseArgs()

    for target in os.listdir('{}/targets'.format(basePath)):
        for assay in os.listdir('{}/targets/{}'.format(basePath, target)):
            if not os.path.isfile('{}/targets/{}/{}/conformations.sdf'.format(basePath, target, assay)):
                continue

            # read data
            r = SDFReader('{}/targets/{}/{}/conformations.sdf'.format(basePath, target, assay), multiconf=True)
            molecules, activities, indizes = [], [], []
            for i, mol in enumerate(r):
                a = extractActivityFromMolecule(mol, args.activityName)
                if not a:
                    continue

                molecules.append(mol)
                activities.append(a)
                indizes.append(i)

            # make sure data set is large enough
            if args.validationSize > 0.5:
                if MIN_SAMPLES_TRAINING_SET * args.nrFolds < len(indizes):
                    continue
            else:
                if MIN_SAMPLES_VALIDATION_SET * args.nrFolds < len(indizes):
                    continue
            #
            # nrValidationSamples = round(len(indizes) * args.validationSize)
            # nrTrainingSamples = len(indizes) - nrValidationSamples

            # split data
            cvFolds = stratifiedCVSplitByActivity(activities, args.nrFolds)

            # save data
            outputDir = '{}/splits/{}/{}'.format(basePath, target, assay)
            if not os.path.isdir(outputDir):
                os.makedirs(outputDir)

            outputName = 'dataSplit-{}-{}'.format(round((1-args.validationSize)*100)/100, args.validationSize) if not args.name else args.name
            with open('{}/{}.json'.format(outputDir, outputName), 'w') as f:
                json.dump(cvFolds, f, indent=2)
