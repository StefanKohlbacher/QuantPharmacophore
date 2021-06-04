# TODO: script to split datasets into training and test data sets.
#  -> evaluate model and parameter selection with CV on training set, then check performance on test set.
#  Keep the established 80-20 vs 20-80 split, but just add test set

# TODO: also run the script and regenerate the tar files

import os
import json
from argparse import ArgumentParser

import numpy as np
from src.molecule_tools import SDFReader
from src.utils import extractActivityFromMolecule


MIN_SAMPLES_TRAINING_SET = 10
MIN_SAMPLES_VALIDATION_SET = 5
MIN_SAMPLES_TEST_SET = 5


def parseArgs():
    parser = ArgumentParser()
    # parser.add_argument('-trainingSize', required=False, default=0.8, type=float)
    parser.add_argument('-validationSize', required=False, default=0.2, type=float)
    parser.add_argument('-testSize', required=False, default=0.1, type=float)
    parser.add_argument('-activityName', required=False, default='pchembl_value', type=str)
    parser.add_argument('-name', required=False, default=None, type=str)
    parser.add_argument('-nrBins', required=False, default=5, type=int)
    parser.add_argument('-nrFolds', required=False, default=5, type=int)
    args = parser.parse_args()
    return args


def stratifiedTrainTestSplitByActivity(samples, activities, nrTestSamples, nrBins, bins=None):
    """
    Put the continuous activity data into bins, which is then used for stratified splitting.
    """
    if not bins:
        # define bins
        bins = {i: [] for i in range(nrBins)}
        minValue, maxValue = min(activities), max(activities)
        binSpan = (maxValue-minValue)/nrBins
        cutoffs = []
        for i in range(1, nrBins+1):
            cutoffs.append(minValue+i*binSpan)

        # populate bins
        for a, s in zip(activities, samples):
            for i, c in enumerate(cutoffs):
                if a > c:  # if never true -> last bin
                    break
            bins[i].append(s)

    # randomly sample test set from bins
    testSet = []
    while True:
        randomBinOrder = np.array(list(bins.keys()))
        np.random.shuffle(randomBinOrder)
        for i in randomBinOrder:
            bin = bins[i]
            if len(bin) == 0:
                bins.pop(i)
                continue
            randomIndex = np.random.choice(np.arange(len(bin)))
            testSet.append(bin[randomIndex])
            del bin[randomIndex]

            if len(testSet) == nrTestSamples or len(bins) == 0:
                break

        if len(testSet) == nrTestSamples or len(bins) == 0:
            break

    # put remaining samples into training set
    trainingSet = [s for s in samples if s not in testSet]
    np.random.shuffle(trainingSet)
    np.random.shuffle(testSet)
    return trainingSet, testSet, bins


def stratifiedCVSplitByActivity(samples, activities, nrFolds, nrValidationSamples, nrBins):
    folds = {}

    # define initial input
    bins = None

    # make folds
    for i in range(nrFolds):
        trainingIndices, testIndices, bins = stratifiedTrainTestSplitByActivity(samples,  # only required the first time
                                                                  activities,  # only required the first time
                                                                  nrValidationSamples,
                                                                  nrBins,
                                                                  bins=bins  # stores all the data needed
                                                                  )
        folds[i] = {'training': trainingIndices, 'validation': testIndices}

    return folds


def splitDataByActivity(indizes, activities, nrValidationSamples, nrTestSamples, nrFolds, nrBins):
    """
    Split data into training and test set. Then split training set into n-fold CV datasets.
    """
    trainingIndices, testIndices, _ = stratifiedTrainTestSplitByActivity(indizes,
                                                                         activities,
                                                                         nrTestSamples,
                                                                         nrBins,
                                                                         )
    # trainingSamples = [indizes[i] for i in trainingIndices]
    trainingActivities = [activities[i] for i in trainingIndices]
    # testSamples = [indizes[i] for i in testIndices]

    folds = stratifiedCVSplitByActivity(trainingIndices, trainingActivities, nrFolds, nrValidationSamples, nrBins)

    return folds, testIndices


if __name__ == '__main__':
    basePath = '../cross_validation'
    args = parseArgs()

    assert args.nrBins >= MIN_SAMPLES_TEST_SET
    assert args.nrBins >= MIN_SAMPLES_VALIDATION_SET

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
            if len(indizes) < MIN_SAMPLES_TEST_SET + MIN_SAMPLES_TRAINING_SET + MIN_SAMPLES_VALIDATION_SET:
                continue

            # nrTestSamples = max(round(len(indizes) * args.testSize), MIN_SAMPLES_TEST_SET)
            # if args.validationSize > 0.5:
            #     nrValidationSamples = max(round(len(indizes) * args.validationSize), MIN_SAMPLES_VALIDATION_SET)
            #     nrTrainingSamples = len(indizes) - nrValidationSamples
            # else:
            #     nrTrainingSamples = len(indizes) - nrTestSamples
            #     nrValidationSamples = max(round(nrTrainingSamples * args.validationSize), MIN_SAMPLES_VALIDATION_SET)
            nrValidationSamples = max(round(len(indizes) * args.validationSize), MIN_SAMPLES_VALIDATION_SET)
            nrTrainingSamples = len(indizes) - nrValidationSamples

            if nrValidationSamples < MIN_SAMPLES_VALIDATION_SET or nrTrainingSamples < MIN_SAMPLES_TRAINING_SET:
                continue

            # split data
            # cvFolds, testSplit = splitDataByActivity(indizes, activities, nrValidationSamples, nrTestSamples, args.nrFolds, args.nrBins)
            cvFolds = stratifiedCVSplitByActivity(indizes, activities, args.nrFolds, nrValidationSamples, args.nrBins)

            # save data
            outputDir = '{}/splits/{}/{}'.format(basePath, target, assay)
            if not os.path.isdir(outputDir):
                os.makedirs(outputDir)

            # dataSplit = {
            #     'cvFolds': cvFolds,
            #     'test': testSplit
            # }

            # outputName = 'dataSplit-{}-{}'.format(args.validationSize, args.testSize) if not args.name else args.name
            outputName = 'dataSplit-{}-{}'.format(round((1-args.validationSize)*100)/100, args.validationSize) if not args.name else args.name
            with open('{}/{}.json'.format(outputDir, outputName), 'w') as f:
                # json.dump(dataSplit, f, indent=2)
                json.dump(cvFolds, f, indent=2)
