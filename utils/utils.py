import numpy as np
import pandas as pd
import CDPL.Chem as Chem
import CDPL.Math as Math
import CDPL.Pharm as Pharm
import matplotlib.pyplot as plt
from Pharmacophore_tools import get_pharmacophore


def bind(instance, func, as_name=None):
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method


def getClosestFeature(queryFeature, referenceFeatures, **kwargs):
    queryType = Pharm.getType(queryFeature)
    queryCoordinates = Chem.get3DCoordinates(queryFeature)
    smallestDistance = 1e3  # we do not assume features to be 1e3 or more Angström apart -> avoids checking for None
    closestFeature = None
    for f in referenceFeatures:
        if queryType != Pharm.getType(f):
            continue

        referenceCoordinates = Chem.get3DCoordinates(f)
        distance = calculateDistance(referenceCoordinates.toArray(), queryCoordinates.toArray())
        if distance < smallestDistance:
            smallestDistance = distance
            closestFeature = f

    return closestFeature, smallestDistance


def getGaussianWeight(query, reference, **kwargs):
    raise NotImplementedError


def getDistanceWeight(distance, maxDistance=1e3, **kwargs):
    distance = np.clip(distance, 1/maxDistance, maxDistance)  # get distance into a reasonable range, so weights don't explode
    return 1/distance


def getFeatureFrequencyWeight(feature, nrSamples, lookupkeys, **kwargs):
    return feature.getProperty(lookupkeys['nrOfFeatures'])/nrSamples


def calculateDistance(coords1, coords2):
    return np.linalg.norm(coords2-coords1, ord=2)


def convertActivityToBinary(a, cutoff=100, logScale=False):
    """
    Reads the activity property of the molecule and converts it to a binary value. The binary value will then replace
    the original activity value.
    :param mol:
    :param cutoff: Cutoff activity value in nM. Everything below will be considered as active -> 1, everything above
    including cutoff as inactive -> 0.
    :param propertyName: Name of property containing the activity.
    :return:
    """
    # from Hyperpharmacophore.features_2 import HYPER_PHARMACOPHORE_PROPERTY_LOOOKUPKEYS


    # a = float(mol.getProperty(HYPER_PHARMACOPHORE_PROPERTY_LOOOKUPKEYS['activity']))

    if logScale:
        # mol.setProperty(HYPER_PHARMACOPHORE_PROPERTY_LOOOKUPKEYS['activity'], 1 if a > cutoff else 0)
        return 1 if a > cutoff else 0
    else:
        # mol.setProperty(HYPER_PHARMACOPHORE_PROPERTY_LOOOKUPKEYS['activity'], 1 if a < cutoff else 0)
        return 1 if a < cutoff else 0
    # return mol


# def consensusVote(samples):
#     values, counts = np.unique(samples, return_counts=True)
#     maxCount = np.argmax(counts)
#     return values[maxCount], counts[maxCount]


def consensusVote(samples, weights=None):
    if weights is None:
        return int(np.round(np.mean(samples)))
    else:
        return int(np.round(np.average(samples, weights=weights)))


def splitTrainTestData(samples, trainingFraction=0.8):
    # extract activities from molecules. Keep most and least active molecules in training set!
    # this makes sure we are only intrapolating and not extraploating to new unseen grounds
    from Hyperpharmacophore.features_2 import HYPER_PHARMACOPHORE_PROPERTY_LOOOKUPKEYS
    from random import shuffle

    activities = [float(s.getProperty(HYPER_PHARMACOPHORE_PROPERTY_LOOOKUPKEYS['activity'])) for s in samples]
    sortedArgs = np.argsort(activities)  # sort ascending order
    samples = [samples[i] for i in sortedArgs]  # sort samples

    nrTrainingSamples = int(np.ceil(len(activities)*trainingFraction))-2
    training = [samples[0]]
    training.append(samples[-1])
    remainingSamples = samples[1: -1]
    shuffle(remainingSamples)  # shuffle the remaining samples again

    training.extend(remainingSamples[:nrTrainingSamples])
    test = remainingSamples[nrTrainingSamples:]
    return training, test


def evaluateActivityMulticonfMolecules(hpModel,
                                       molecules,
                                       aggregationFn=np.average,  # use average as default instead of mean -> takes weight argument
                                       weightedConformations=False,
                                       kBestConformations=None,
                                       ):
    from Pharmacophore_tools import get_pharmacophore

    activities = []
    for mol in molecules:
        confActivities = []
        confScores = []
        for j in range(Chem.getNumConformations(mol)):
            Chem.applyConformation(mol, j)
            ph4 = get_pharmacophore(mol)
            _, confActivity, score = hpModel.predict(ph4, returnScore=True)[0]
            if score > 0:
                if isinstance(confActivity, np.ndarray):
                    confActivities.append(confActivity[0])
                else:
                    confActivities.append(confActivity)
                confScores.append(score)
        if len(confActivities) == 0:
            activities.append(0)
            continue
        if weightedConformations:
            if kBestConformations is not None:
                sortedArgs = np.argsort(confScores)
                confActivities = [confActivities[i] for i in reversed(sortedArgs)]
                confScores = [confScores[i] for i in reversed(sortedArgs)]
                if kBestConformations+1 >= len(sortedArgs):  # more conformations than required
                    confActivities = confActivities[:kBestConformations+1]  # +1 one because last conf will have weight 0
                    confScores = confScores[:kBestConformations+1]

            if len(set(confScores)) > 1:
                weights = minMaxScale(confScores)
            else:
                weights = [1]*len(confScores)
            value = aggregationFn(confActivities, weights=weights)

        elif kBestConformations is not None:
            sortedArgs = np.argsort(confScores)
            confActivities = [confActivities[i] for i in reversed(sortedArgs)]
            if kBestConformations >= len(sortedArgs):
                confActivities = confActivities[:kBestConformations]
            value = aggregationFn(confActivities)
        else:
            value = aggregationFn(confActivities)
        activities.append(round(value, 3))
    return activities


def evaluateBinaryMulticonfMolecules(hpModel,
                                     molecules,
                                     weightedConformations=False,
                                     kBestConformations=None,
                                     ):
    return evaluateActivityMulticonfMolecules(hpModel,
                                              molecules,
                                              aggregationFn=consensusVote,
                                              weightedConformations=weightedConformations,
                                              kBestConformations=kBestConformations,
                                              )


def make_activity_plot(y_true, y_pred, xLabel='true values', yLabel='predicted values'):

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true.flatten(), y_pred.flatten())
    # define axis_limits
    high_activity_lim = np.ceil(max(max(y_true), max(y_pred)))
    low_activity_lim = np.floor(min(min(y_true), min(y_pred)))
    limits = (high_activity_lim, low_activity_lim)
    # limits = (low_activity_lim, high_activity_lim)

    # add regression line
    m, b = np.polyfit(y_true.flatten(), y_pred.flatten(), 1)
    x = np.arange(low_activity_lim, high_activity_lim+1)
    ax.plot(x, m*x+b)

    # set limits on plot
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    return fig, ax


def makeBoxplot(values):
    fig, ax = plt.subplots()
    ax.set_title('Errors')
    ax.boxplot(values)
    return fig, ax


def minMaxScale(values):
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    minValue, maxValue = np.min(values), np.max(values)
    # assert len(values) > 1, 'Values need to contain at least two entities to be scaled'
    # if minValue == maxValue:
    #     print('Something')
    assert maxValue > minValue, 'Max needs to be bigger than min, {maxValue} {minValue} {values}'.format(minValue=minValue, maxValue=maxValue, values=values)
    # assert maxValue != 0, 'Max value needs to be bigger than 0'
    normalized = (values - minValue) / (maxValue-minValue)
    return normalized


def standardizeActivityUnits(activity, unit):
    if unit == 'M':
        factor = 1
    elif unit == 'mM':
        factor = 1e-3
    elif unit == 'µM':
        factor = 1e-6
    elif unit == 'nM':
        factor = 1e-9
    else:
        raise ValueError('Activity unit %s not known. Use one of [nM, µM, mM, M]' % unit)
    return activity * factor


def extractActivityFromMolecule(mol, activityProp):
    dataBlock = Chem.getStructureData(mol)
    for p in dataBlock:
        if activityProp in p.header:
            if p.data == 'nan':
                return None
            else:
                return float(p.data)


def numFeaturesBaseline(trainingSet, testSet, activityLookupKey, model=None, returnPredictions=False):
    if model is None:
        from sklearn.linear_model import Ridge

        model = Ridge(fit_intercept=True)
    from ML_tools import analyse_regression

    numFeatures = []
    activities = []
    for sample in trainingSet:
        if isinstance(sample, Chem.BasicMolecule):
            p = get_pharmacophore(sample)
            p.setProperty(activityLookupKey, sample.getProperty(activityLookupKey))
            sample = p
        activities.append(sample.getProperty(activityLookupKey))
        numFeatures.append(sample.numFeatures)

    predictionFeatures = []
    trueActivities = []
    for sample in testSet:
        if isinstance(sample, Chem.BasicMolecule):
            p = get_pharmacophore(sample)
            p.setProperty(activityLookupKey, sample.getProperty(activityLookupKey))
            sample = p
        trueActivities.append(sample.getProperty(activityLookupKey))
        predictionFeatures.append(sample.numFeatures)

    model.fit(np.array(numFeatures).reshape(-1, 1), np.array(activities))
    predictions = model.predict(np.array(predictionFeatures).reshape(-1, 1))

    if returnPredictions:
        return analyse_regression(np.array(trueActivities), predictions.flatten()), predictions.flatten()

    return analyse_regression(np.array(trueActivities), predictions.flatten())


def standardPropertiesBaseline(trainingSet, testSet, activityLookupKey, model=None, returnPredictions=False):
    if model is None:
        from sklearn.linear_model import Ridge

        model = Ridge(fit_intercept=True)
    from Molecule_tools import calculateStandardProperties
    from ML_tools import analyse_regression

    # extract training properties
    standardProperties = calculateStandardProperties(trainingSet)
    activities = []
    for sample in trainingSet:
        activities.append(sample.getProperty(activityLookupKey))
    standardProperties = pd.DataFrame.from_dict(standardProperties, orient='columns').values

    # extract test properties
    testProps, testActivities = calculateStandardProperties(testSet), []
    for sample in testSet:
        testActivities.append(sample.getProperty(activityLookupKey))
    testProps = pd.DataFrame.from_dict(testProps, orient='columns').values

    # fit model
    model.fit(np.stack(standardProperties, axis=0), np.array(activities))
    predictions = model.predict(np.stack(testProps, axis=0))

    if returnPredictions:
        return analyse_regression(np.array(testActivities), predictions.flatten()), predictions.flatten()
    return analyse_regression(np.array(testActivities), predictions.flatten())


def pharmacophoreFingerprintBaseline(trainingSet, testSet, activityLookupKey):
    raise NotImplementedError


def runTimeHandler(signum, frame):
    message = 'Caught signal -> time ran out!'
    print(message)
    raise TimeoutError(message)


def softmax(array):
    exp = np.exp(array)
    return exp / np.sum(exp)


def trainValidateTestModel(model, args):
    """
    Load dataset, train model, validate model and then choose best model from validation to test on test set.

    Saves results and plots to specified path.
    :param model:
    :param args:
    :return:
    """
    import json

    with open('{}trainValidationTestSplit.json'.format(args.dataset), 'r') as f:
        trainTestValidationSplit = json.load(f)


def selectMostRigidMolecule(molecules):
    """
    Determine most rigid / least flexible molecule based on number of single non-hydrogen bonds in a molecule.
    :param molecules:
    :return:
    """
    numberOfFlexibleBondsPerMolecule = []
    for mol in molecules:
        singleNonHydrogenBonds = 0
        for bond in mol.bonds:
            if Chem.getOrder(bond) != 1:
                continue

            beginAtom, endAtom = bond.getBegin(), bond.getEnd()
            if Chem.getType(beginAtom) != 1 and Chem.getType(endAtom) != 1:
                singleNonHydrogenBonds += 1

        numberOfFlexibleBondsPerMolecule.append(singleNonHydrogenBonds)

    mostRigidMolecule = np.argmin(numberOfFlexibleBondsPerMolecule)
    return molecules[mostRigidMolecule], [molecules[i] for i in range(len(molecules)) if i != mostRigidMolecule]


class ParamsHoldingClass:

    def __init__(self, params):
        self._params = params
        for key, value in params.items():
            setattr(self, key, value)

    def __iter__(self):
        for key, value in self._params.items():
            yield key, value


class AlignmentError(Exception):

    def __init__(self, message=None):
        if message is None:
            message = 'No alignment found.'
        super(AlignmentError, self).__init__(message)
