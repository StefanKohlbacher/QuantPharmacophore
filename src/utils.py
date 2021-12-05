import numpy as np
import pandas as pd
import CDPL.Chem as Chem
import CDPL.Pharm as Pharm
import matplotlib.pyplot as plt
from src.pharmacophore_tools import getPharmacophore
from sklearn.metrics import mean_squared_error, r2_score


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


def make_activity_plot(y_true, y_pred, xLabel='true values', yLabel='predicted values'):

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true.flatten(), y_pred.flatten())
    # define axis_limits
    high_activity_lim = np.ceil(max(max(y_true), max(y_pred)))
    low_activity_lim = np.floor(min(min(y_true), min(y_pred)))
    limits = (high_activity_lim, low_activity_lim)
    #add parameters in graph
    high_annotate_lim= high_activity_lim -0.5
    low_annotate_lim = low_activity_lim + 0.2
    ax.annotate("R-squared = {:.3f}\nRMSE = {}".format(r2_score(y_true, y_pred),mean_squared_error(y_true,y_pred,squared=False)), (low_annotate_lim, high_annotate_lim))
    
    limits = (low_activity_lim, high_activity_lim)

    # add regression line
    m, b = np.polyfit(y_true.flatten(), y_pred.flatten(), 1)
    x = np.arange(low_activity_lim, high_activity_lim+1)
    ax.plot(x, m*x+b, color='r')

    # set limits on plot
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    return fig, ax


def extractActivityFromMolecule(mol, activityProp):
    dataBlock = Chem.getStructureData(mol)
    for p in dataBlock:
        if activityProp in p.header:
            if p.data == 'nan':
                return None
            else:
                try:
                    return float(p.data)
                except:
                    return None


def numFeaturesBaseline(trainingSet, testSet, activityLookupKey, model=None, returnPredictions=False):
    if model is None:
        from sklearn.linear_model import Ridge

        model = Ridge(fit_intercept=True)
    from src.ml_tools import analyse_regression

    numFeatures = []
    activities = []
    for sample in trainingSet:
        if isinstance(sample, Chem.BasicMolecule):
            p = getPharmacophore(sample)
            p.setProperty(activityLookupKey, sample.getProperty(activityLookupKey))
            sample = p
        activities.append(sample.getProperty(activityLookupKey))
        numFeatures.append(sample.numFeatures)

    predictionFeatures = []
    trueActivities = []
    for sample in testSet:
        if isinstance(sample, Chem.BasicMolecule):
            p = getPharmacophore(sample)
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
    from src.molecule_tools import calculateStandardProperties
    from src.ml_tools import analyse_regression

    # extract training properties
    standardProperties = calculateStandardProperties(trainingSet)
    activities = []
    for sample in trainingSet:
        activities.append(sample.getProperty(activityLookupKey))
    standardProperties = pd.DataFrame.from_dict(standardProperties, orient='columns').values

    # extract _test properties
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


def runTimeHandler(signum, frame):
    message = 'Caught signal -> time ran out!'
    print(message)
    raise TimeoutError(message)


def selectMostRigidMolecule(molecules, returnIndices=False):
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

    try:
        mostRigidMolecule = np.argmin(numberOfFlexibleBondsPerMolecule)
    except IndexError:
        mostRigidMolecule = 0
    if returnIndices:
        return mostRigidMolecule, [k for k in range(len(molecules)) if k != mostRigidMolecule]
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
