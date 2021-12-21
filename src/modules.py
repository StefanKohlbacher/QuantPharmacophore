from typing import List

import CDPL.Chem as Chem
import CDPL.Pharm as Pharm
import numpy as np
import pandas as pd
import sys
import json
from src.qphar import LOOKUPKEYS, Qphar
from src.utils import AlignmentError
from src.ml_tools import analyse_regression


REQUIREMENTS = {
    'activityRange': 3,
    'boxplot_range': 2.5,
    'kld': 0.75,
}


DEFAULT_TRAINING_PARAMETERS = {
    'fuzzy': True,
    'weightType': 'distance',
    'modelType': 'randomForest',
    'modelKwargs': {'n_estimators': 10, 'max_depth': 3},
    'threshold': 1.5,
    'mergeOnce': True,
    'metric': 'R2'
}

DEFAULT_MODEL_PARAMETERS = {
    'randomForest': {'n_estimators': 10, 'max_depth': 3},
    'ridge': {'fit_intercept': False},
    'pca_ridge': {},
    'pls': {},
    'pca_lr': {}
}


def kullbackLeiblerDivergence(p, q):
    """
    0 indicates the two distributions are the same. There are no upper limits to the KL-Divergence.
    """
    assert len(p) == len(q)
    aggregator = 0
    for i in range(len(p)):
        if p[i] == 0:
            continue
        aggregator += p[i]*np.log(p[i]/q[i])
    return aggregator


def estimateKLDivergence(activities):
    if not isinstance(activities, np.ndarray):
        activities = np.array(activities)

    np.sort(activities)

    # calculate the kullback-leibler divergence for a given dataset
    uniform = {
        'min': min(activities),
        'max': max(activities) + 0.001,  # add a very small amount so we include the empirical values later on
        'stepSize': (max(activities) - min(activities)) / (len(activities) - 1),
        'probability': 1 / len(activities)
    }

    # first we need to estimate the distribution of the empirical values given a discrete interval
    nrIntervals = len(activities)

    samplesPerInterval = []
    for step in range(nrIntervals):
        lowerEnd = uniform['min'] + step * uniform['stepSize']
        upperEnd = uniform['min'] + (step + 1) * uniform['stepSize']
        largerThanLower = (activities >= lowerEnd)
        smallerThanUpper = (activities < upperEnd)
        nrS = np.sum(largerThanLower * smallerThanUpper)
        samplesPerInterval.append(nrS)

    probabilities = np.array(samplesPerInterval) / nrIntervals
    uniformValues = np.repeat(uniform['probability'], len(probabilities))

    kld = kullbackLeiblerDivergence(probabilities, uniformValues)
    return kld


def splitData(molecules, activityName, validationFraction=None, testFraction=None):
    """
    Split data into training, validation [optional], _test [optional] set. The least and most active compounds are
    retained in the training set.
    If both _test and validation fraction are given, then _test fraction determines size of training set including
    validation set. The validation set is then taken in proportion from the total training set.
    I.e.
    testFraction = 0.2; validationFraction = 0.1
    --> trainingFraction = (1 - testFraction) * validationFraction


    :param molecules:
    :param activityName:
    :param validationFraction:
    :param testFraction:
    :return:
    """
    molecules, activities = splitSamplesActivities(molecules, activityName)
    sortedArgs: List[int] = np.argsort(activities).tolist()
    # add most and least active compound to training set
    trainingIndices, validationIndices, testIndices = [sortedArgs[0]], [], []
    trainingIndices.append(sortedArgs[-1])
    del sortedArgs[0], sortedArgs[-1]

    if validationFraction is None and testFraction is None:
        print('Fraction of validation or _test set not specified. Training set is kept as it is.')
        return {'trainingSet': molecules, 'validationSet': [], 'testSet': []}

    if validationFraction is None:  # just train-_test split
        trainingFraction = 1-testFraction
        print('Training set: {} %'.format(trainingFraction*100))
        print('Test set: {} %'.format(testFraction * 100))
        nrTestSamples = int(np.floor(len(activities) * testFraction))

        while len(testIndices) < nrTestSamples:  # do the inverse
            i = np.random.choice(np.arange(len(sortedArgs)))
            testIndices.append(sortedArgs[i])
            del sortedArgs[i]

        trainingIndices.extend(sortedArgs)

    elif testFraction is None:  # same as above but label as validation set
        trainingFraction = 1 - validationFraction
        print('Training set: {} %'.format(trainingFraction * 100))
        print('Validation set: {} %'.format(validationFraction * 100))
        nrValidationSamples = int(np.floor(len(activities) * validationFraction))

        while len(validationIndices) < nrValidationSamples:  # do the inverse
            i = np.random.choice(np.arange(len(sortedArgs)))
            validationIndices.append(sortedArgs[i])
            del sortedArgs[i]

        trainingIndices.extend(sortedArgs)

    else:  # both are not None --> train-validation-_test split
        trainingFraction = (1 - testFraction) - (1 - testFraction) * validationFraction
        print('Training set: {} %'.format(trainingFraction * 100))
        print('Validation set: {} %'.format((1 - testFraction) * validationFraction * 100))
        print('Test set: {} %'.format(testFraction * 100))
        nrTestSamples = int(np.floor(len(activities) * testFraction))
        nrValidationSamples = int(np.floor((len(activities) - nrTestSamples) * validationFraction))

        while len(testIndices) < nrTestSamples:  # do the inverse
            i = np.random.choice(np.arange(len(sortedArgs)))
            testIndices.append(sortedArgs[i])
            del sortedArgs[i]

        while len(validationIndices) < nrValidationSamples:
            i = np.random.choice(np.arange(len(sortedArgs)))
            validationIndices.append(sortedArgs[i])
            del sortedArgs[i]

        trainingIndices.extend(sortedArgs)

    trainingSet = [molecules[i] for i in trainingIndices]
    validationSet = [molecules[i] for i in validationIndices]
    testSet = [molecules[i] for i in testIndices]
    return {'trainingSet': trainingSet, 'validationSet': validationSet, 'testSet': testSet}


def splitSamplesActivities(samples, activityName):
    from src.utils import extractActivityFromMolecule

    molecules, activities = [], []
    for i, mol in enumerate(samples):
        a = extractActivityFromMolecule(mol, activityName)
        molName = Chem.getName(mol)
        try:
            a = float(a)
            mol.setProperty(LOOKUPKEYS['activity'], a)

            hasCoordinates = checkCoordinates(mol)
            if not hasCoordinates:
                print(
                    'Molecule {} with name {} has no 3D coordinates. Please make sure given molecules have at least one 3D coordinate set. Skipping'.format(
                        i, molName))
                continue

            molecules.append(mol)
            activities.append(a)
        except:
            print('Activity of molecule {} with name {} cannot be parsed. Please make sure it is a number'.format(i,molName))

    return molecules, activities


def addPropertyToSDFData(molecule, key, value):
    sd = Chem.getStructureData(molecule)
    sd.addEntry(' <{}>'.format(key), str(value))
    Chem.setStructureData(molecule, sd)
    return molecule


def checkDataRequirements(samples, activityName=None):
    import matplotlib.pyplot as plt

    if activityName is None:  # _test/prediction data --> just check whether we have coordinates
        for i, sample in enumerate(samples):
            hasCoordinates = checkCoordinates(sample)
            if not hasCoordinates:
                print('Sample {} has no 3D coordinates. Please generate 3D coordinates'.format(i))
        print('All samples have 3D coordinates and can be used to predict activities')
        return

    molecules, activities = splitSamplesActivities(samples, activityName)

    # check if activities fulfill requirements
    if len(activities) == 0:
        print('No molecules left for checking data.')
        print('Exiting')
        sys.exit()
    activityRange = max(activities) - min(activities)
    if activityRange < REQUIREMENTS['activityRange']:
        print(
            'Available samples do not span a large enough activity range to make a proper QSAR study. It is recommended to either enlarge the dataset or pick a different dataset.')
        return False

    # check activity range ignoring outliers
    bb = plt.boxplot(activities)
    lower, upper = bb['whiskers'][0], bb['whiskers'][1]
    activityRange = upper.get_ydata()[1] - lower.get_ydata()[0]
    if activityRange < REQUIREMENTS['boxplot_range']:
        print(
            'Available samples do not span a learge enough activity range, ignoring the outliers, to make a proper QSAR study. It is recommended to either enlarge the dataset or pick a different dataset.')
        return False

    # filter by KL-divergence. we would like the data to be evenly distributed and not clustered into a certain region --> similar to uniform distribution
    kld = estimateKLDivergence(activities)
    if kld > REQUIREMENTS['kld']:
        print(
            'The given dataset seems to cluster in some region of activity value. This introduces a bias to the model.')
        print('It is not recommended to pursue a QSAR with this dataset.')
        return False

    print('The dataset fulfills all the requirements to perform QSAR.')
    return True


def hasNonZeroCoordinates(entity3d):
    coords = Chem.get3DCoordinatesArray(entity3d).toArray(False)
    if np.any(coords):
        return True
    return False


def checkCoordinates(sample):
    if isinstance(sample, Chem.BasicMolecule):

        for a in sample.atoms:
            if not Chem.has3DCoordinates(a):
                return False
            if hasNonZeroCoordinates(a):
                return True  # at least one atom has non-zero coordinates --> molecule has a valid 3D structure

    elif isinstance(sample, Pharm.BasicPharmacophore):
        for f in sample:
            if not Chem.has3DCoordinates(f):
                return False
            if hasNonZeroCoordinates(f):
                return True

    else:
        print('Given sample of type {} not recognized. Please provide a molecule or a pharmacophore'.format(type(sample)))
        return False

    return False


def makeTrainingRun(molecules, activities, parameters):
    """
    Trains a model based on given datasets and parameters.

    Returns the trained model, performance on the training set and predictions on the training set.
    :param molecules:
    :param activities:
    :param parameters:
    :return:
    """
    from src.utils import selectMostRigidMolecule

    # prepare data
    templateIndex, remainingMoleculesIndices = selectMostRigidMolecule(molecules, returnIndices=True)
    template = molecules[templateIndex]
    remainingMolecules = [molecules[k] for k in remainingMoleculesIndices]
    reorderdedActivities = [activities[templateIndex]]
    reorderdedActivities.extend([activities[k] for k in remainingMoleculesIndices])
    trainingSet = [template]
    trainingSet.extend(remainingMolecules)

    # train
    models, predictions = train(template, remainingMolecules, parameters)  # predictions are [template, *remaining]

    # evaluate and select best model
    performance = {}
    for i, y_pred in enumerate(predictions):
        performance[i] = analyse_regression(np.array(reorderdedActivities), y_pred)
    df = pd.DataFrame.from_dict(performance, orient='index')
    df.sort_values(parameters.get('metric', DEFAULT_TRAINING_PARAMETERS['metric']), inplace=True, ascending=False)
    bestModelIndex = df.index.values[0]

    # assign predictions to molecules
    for mol, pred in zip(trainingSet, predictions[bestModelIndex]):
        addPropertyToSDFData(mol, 'prediction', pred)
        mol.setProperty(LOOKUPKEYS['prediction'], pred)

    return models[bestModelIndex], performance[bestModelIndex], trainingSet


def makeTrainingTestRun(trainingMolecules, trainingActivities, testMolecules, testActivities, parameters):
    """
    Trains model on training set and evaluates it on the _test set. Best model along with its predictions are returned.
    :param trainingMolecules:
    :param trainingActivities:
    :param testMolecules:
    :param testActivities:
    :param parameters:
    :return:
    """
    from src.utils import selectMostRigidMolecule

    # prepare data
    templateIndex, remainingMoleculesIndices = selectMostRigidMolecule(trainingMolecules, returnIndices=True)
    template = trainingMolecules[templateIndex]
    remainingMolecules = [trainingMolecules[k] for k in remainingMoleculesIndices]
    reorderdedActivities = [trainingActivities[templateIndex]]
    reorderdedActivities.extend([trainingActivities[k] for k in remainingMoleculesIndices])
    trainingSet = [template]
    trainingSet.extend(remainingMolecules)

    # train
    models, trainingpredictions = train(template, remainingMolecules, parameters)  # predictions are [template, *remaining]

    # evaluate and select based model based on _test set
    testPerformance = {}
    testPredictions = {}
    for i, model in enumerate(models):
        y_pred_test = model.predict(testMolecules)
        testPerformance[i] = analyse_regression(np.array(testActivities), y_pred_test)
        testPredictions[i] = y_pred_test
    df = pd.DataFrame.from_dict(testPerformance, orient='index')
    df.sort_values(parameters.get('metric', DEFAULT_TRAINING_PARAMETERS['metric']), inplace=True, ascending=False)
    bestModelIndex = df.index.values[0]

    # evaluate training performance
    trainingPerformance = analyse_regression(reorderdedActivities, trainingpredictions[bestModelIndex])

    # assign predictions to molecules
    for mol, y_pred in zip(trainingSet, trainingpredictions[bestModelIndex]):
        addPropertyToSDFData(mol, 'prediction', y_pred)  # property is added in place
        mol.setProperty(LOOKUPKEYS['prediction'], y_pred)
    for mol, y_pred in zip(testMolecules, testPredictions[bestModelIndex]):
        addPropertyToSDFData(mol, 'prediction', y_pred)
        mol.setProperty(LOOKUPKEYS['prediction'], y_pred)

    return models[bestModelIndex], trainingPerformance, trainingSet, testPerformance[bestModelIndex], testMolecules


def train(template, remainingSamples, params):
    models, predictions = [], []
    trainingSet = [template]
    trainingSet.extend(remainingSamples)
    for j in range(len(remainingSamples)):
        try:
            model = Qphar([template, remainingSamples[j]],
                          **{k: v for k, v in params.items() if k != 'logPath'})
        except AlignmentError:
            continue

        model.fit([remainingSamples[k] for k in range(len(remainingSamples)) if k != j],
                  mergeOnce=params.get('mergeOnce', DEFAULT_TRAINING_PARAMETERS['mergeOnce']),
                  threshold=params.get('threshold', DEFAULT_TRAINING_PARAMETERS['threshold']))
        models.append(model)
        predictions.append(model.predict(trainingSet))

    return models, predictions


def gridSearch(datasets, searchParams, nrProcesses=1, outputPath=None):
    """

    :param datasets: Dictionary associating training, validation and testset with their paths. Datasets are loaded
    directly in the execution function --> allows us to schedule jobs run in parallel --> molecules cannot be pickled.
    :param searchParams:
    :return:
    """

    from itertools import product
    import os
    from src.utils import make_activity_plot
    import matplotlib.pyplot as plt

    # create folder where all results are saved to
    if outputPath is None:
        outputPath = './gridSearch'

    if outputPath.endswith('/'):
        outputPath = outputPath[:-1]

    i = 0
    while os.path.isdir('{}_{}'.format(outputPath, i)):
        i += 1
    outputPath = '{}_{}/'.format(outputPath, i)

    # make parameter combinations to search
    keys = sorted(searchParams.keys())
    combinations = product(*(searchParams[k] for k in keys))
    combinedParameters = [{keys[k]: values[k] for k in range(len(keys))} for values in combinations]  # all parameters

    # schedule jobs
    jobs = []
    for i, params in enumerate(combinedParameters):
        allParams = {k: v for k, v in DEFAULT_TRAINING_PARAMETERS.items()}
        for k, v in params.items():
            allParams[k] = v
        allParams['modelKwargs'] = DEFAULT_MODEL_PARAMETERS.get(params['modelType'], {})
        allParams['i'] = len(jobs)

        jobs.append((datasets, allParams, outputPath, i))

    # train models in parallel and evaluate on validation set. Save results, models and predictions.
    runParallel(executeTrainingValidation, jobs, nrProcesses=nrProcesses)

    # for each run, load the saved models and _test them on the _test set --> determine best model overall
    testSet = loadMolecules(datasets['testSet'])
    testMolecules, testActivities = splitSamplesActivities(testSet, 'activity')

    modelPerformances = {}
    for args in jobs:
        i = args[-1]
        path = '{}{}/'.format(outputPath, i)
        model = Qphar()
        model.load('{}model/'.format(path))
        predictions = []
        for mol in testMolecules:
            pred = predict(model, [mol])
            mol.setProperty(LOOKUPKEYS['prediction'], pred)
            predictions.append(pred)
        predictions = np.array(predictions)
        performance = analyse_regression(np.array(testActivities), predictions)
        modelPerformances[i] = performance

        # save predictions and performance as well as plots
        fig, _ = make_activity_plot(np.array(testActivities), np.array(predictions), xLabel='p(activity) true',
                                    yLabel='p(activity) predicted')
        fig.savefig('{}testPredictions.png'.format(path))
        plt.close()
        savePerformance(performance, '{}testPerformance'.format(path))
        for mol, y_pred in zip(testMolecules, predictions):
            addPropertyToSDFData(mol, 'prediction', y_pred)
            mol.setProperty(LOOKUPKEYS['prediction'], y_pred)
        saveMolecules(testMolecules, '{}testPredictions.sdf'.format(path))

    # save aggregated _test results
    df = pd.DataFrame.from_dict(modelPerformances, orient='index')
    df.to_csv('{}gridSearchResults.csv'.format(outputPath))
    return df


def executeTrainingValidation(datasets, parameters, outputPath, jobNr):
    import os

    # load molecules
    trainingSet = loadMolecules(datasets['trainingSet'])
    validationSet = loadMolecules(datasets['validationSet'])
    trainingMolecules, trainingActivities = splitSamplesActivities(trainingSet, 'activity')
    validationMolecules, validationActivities = splitSamplesActivities(validationSet, 'activity')

    # train and evaluate on validation set
    model, trainingPerformance, trainingMolecules, validationPerformance, validationMolecules = makeTrainingTestRun(trainingMolecules,
                                                                                                                    trainingActivities,
                                                                                                                    validationMolecules,
                                                                                                                    validationActivities,
                                                                                                                    parameters)

    # save models
    if not os.path.isdir('{}{}/'.format(outputPath, jobNr)):
        os.makedirs('{}{}/'.format(outputPath, jobNr))

    model.save('{}{}/model/'.format(outputPath, jobNr))
    saveMolecules(trainingMolecules, '{}{}/trainingPredictions.sdf'.format(outputPath, jobNr))
    savePerformance(trainingPerformance, '{}{}/trainingPerformance'.format(outputPath, jobNr))
    plotPredictionsFromMolecules(trainingMolecules, '{}{}/training.png'.format(outputPath, jobNr))
    saveMolecules(validationMolecules, '{}{}/validationPredictions.sdf'.format(outputPath, jobNr))
    savePerformance(validationPerformance, '{}{}/validationPerformance'.format(outputPath, jobNr))
    plotPredictionsFromMolecules(validationMolecules, '{}{}/validation.png'.format(outputPath, jobNr))


def predict(model, samples, **kwargs):
    predictions = model.predict(samples)
    return predictions


def loadParams(path):
    import json

    with open(path, 'r') as f:
        params = json.load(f)

    return params


def loadMolecules(path, multiconf=True):
    from src.molecule_tools import SDFReader

    r = SDFReader(path, multiconf=multiconf)
    molecules = [mol for mol in r]
    if len(molecules) == 0:
        print('No molecules loaded. Did you specify the correct path?')
        print('Exiting')
        sys.exit()
    return molecules


def saveMolecules(molecules, path, multiconf=True):
    from src.molecule_tools import mol_to_sdf

    # add activity property if present, so we can access it later on by the known name
    for mol in molecules:
        sd = Chem.getStructureData(mol)
        headers = [p.header for p in sd]
        if 'activity' in headers:
            continue
        a = mol.getProperty(LOOKUPKEYS['activity'])
        sd.addEntry(' <activity>', str(a))
        Chem.setStructureData(mol, sd)

    mol_to_sdf(molecules, path, multiconf=multiconf)


def savePerformance(performance, path):
    if not isinstance(performance, pd.DataFrame):
        for key in list(performance.keys()):
            if isinstance(performance[key], np.integer): 
                performance[key] = int(performance[key])
            elif isinstance(performance[key], np.floating):
                performance[key] = float(performance[key])
            elif isinstance(performance[key], np.ndarray): 
                performance[key] = performance[key].tolist()

        with open('{}.json'.format(path), 'w') as f:
            json.dump(performance, f)

    else:
        performance.to_csv('{}.csv'.format(path))


def runParallel(targetFn, jobs, nrProcesses=2):
    if nrProcesses > 1:
        import multiprocessing as mp

        pool = mp.Pool(nrProcesses)
        scheduledJobs = []
        for job in jobs:
            scheduledJobs.append(pool.apply_async(targetFn, args=job))

        for job in scheduledJobs:
            job.get()

        pool.close()

    else:
        for job in jobs:
            targetFn(*job)


def plotPredictionsFromMolecules(molecules, path):
    from src.utils import make_activity_plot
    import matplotlib.pyplot as plt

    y_true, y_pred = [], []
    for mol in molecules:
        y_true.append(mol.getProperty(LOOKUPKEYS['activity']))
        y_pred.append(mol.getProperty(LOOKUPKEYS['prediction']))

    fig, _ = make_activity_plot(np.array(y_true), np.array(y_pred), xLabel='p(activity) true',
                                yLabel='p(activity) predicted')
    fig.savefig(path)
    plt.close()
