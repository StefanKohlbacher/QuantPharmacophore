import CDPL.Pharm as Pharm
import CDPL.Chem as Chem
import CDPL.Base as Base
import CDPL.Math as Math
import os
import json
import time
import numpy as np
from collections.abc import Iterable
from abc import abstractmethod
import signal
from src.utils import runTimeHandler, AlignmentError, getClosestFeature, getDistanceWeight, getGaussianWeight, calculateDistance, getFeatureFrequencyWeight
from src.pharmacophore_tools import getPharmacophore, savePharmacophore, loadPharmacophore


signal.signal(signal.SIGALRM, runTimeHandler)


LOOKUPKEYS = {
    'activity': Base.LookupKey.create('activity'),
    'activities': Base.LookupKey.create('activities'),
    'nrOfFeatures': Base.LookupKey.create('nrOfFeatures'),
    'prediction': Base.LookupKey.create('prediction'),
}


def assignActivitiesToMolecules(molecules, activities):
    for mol, a in zip(molecules, activities):
        mol.setProperty(LOOKUPKEYS['activity'], float(a))
        mol.setProperty(LOOKUPKEYS['activities'], [float(a)])


def assignActivitiesToPharmacophores(pharmacophores, activities):
    for p, a in zip(pharmacophores, activities):
        p.setProperty(LOOKUPKEYS['activity'], float(a))

        for feature in p:
            feature.setProperty(LOOKUPKEYS['activity'], float(a))
            feature.setProperty(LOOKUPKEYS['activities'], [float(a)])
            feature.setProperty(LOOKUPKEYS['nrOfFeatures'], 1)


class HyperPharmacophore(Pharm.BasicPharmacophore):

    """
    This class is an extended version of the inital Hyperpharmacophore class. Changes are
    implemented as discussed in the project journal and according to feedback gained during the progress report.

    The main change is the type of alignment and feature creation. Samples are aligned to a given template and all its
    features are added to the HP at first. Then, once all samples are aligned to the template, the features are
    clustered and representative features of the set are chosen as features for the HP.
    """

    def __init__(self,
                 template=None,
                 modelType='linearRegression',
                 modelKwargs=None,
                 logPath=None,
                 weightType='distance',
                 name='Hyperpharmacophore',
                 alignmentTimeout=30,
                 fuzzy=True,
                 threshold=1.5,
                 **kwargs):
        super(HyperPharmacophore, self).__init__()
        self.template = template
        self.modelType = modelType
        self.modelKwargs = {} if modelKwargs is None else modelKwargs
        self.logPath = logPath
        self.weightType = weightType
        self.name = name
        self.timeout = alignmentTimeout
        self.fuzzy = fuzzy
        self.threshold = threshold
        self.kwargs = kwargs

        # set up pharmacophore aligner
        self.aligner = Pharm.PharmacophoreAlignment(True)
        self.scorer = Pharm.PharmacophoreFitScore()

        # set up ml model
        if isinstance(self.modelType, str):
            self.mlModel = self._initMLModel(self.modelType, self.modelKwargs)
        elif isinstance(self.modelType, Iterable):
            models = []
            for mType in self.modelType:
                m = self._initMLModel(mType, self.modelKwargs[mType])
                models.append(m)
            self.mlModel = models
        else:
            raise TypeError('Unrecognized modelType {}. Should be either a str or a list strings'.format(type(self.modelType)))

        # some collections to store data
        self.samples = []  # contains initial samples, without any specific conformation
        self.alignedSamples = []  # contains tuples of aligned pharmacophores and scores
        self.cleanedHP = Pharm.BasicPharmacophore()
        self.nrSamples = 0

        # init template
        if self.template is None:
            return

        if isinstance(template, Pharm.BasicPharmacophore):
            self.nrSamples += 1
            for f in template:
                newFeature = self.addFeature()
                newFeature.assign(f)
            self.samples.append(template)
            self.alignedSamples.append((template, 0))

        elif isinstance(template, Iterable) and len(template) == 2:  # assume molecules
            m1, m2 = template[0], template[1]
            if not isinstance(m1, Chem.BasicMolecule) or not isinstance(m2, Chem.BasicMolecule):
                raise TypeError('If template is iterable, length needs to be 2, whereas both entities need to be of type Chem.BasicMolecule')

            template, firstSample, score = self._initFromMolecules(m1, m2)
            for f in template:
                newFeature = self.addFeature()
                newFeature.assign(f)
            self.nrSamples += 2
            self.samples.extend([m1, m2])
            self.alignedSamples.extend([(template, score), (firstSample, score)])

        else:
            raise TypeError('Given type {} not known. Type should be one of [Chem.BasicMolecule, Pharm.BasicPharmacophore].'.format(type(template)))

    def _initMLModel(self, modelType, modelKwargs):

        # set up ml model
        if modelType == 'linearRegression':
            from sklearn.linear_model import LinearRegression

            return LinearRegression(**getattr(self, 'modelKwargs', {}))
        elif modelType == 'ridge':
            from sklearn.linear_model import Ridge

            return Ridge(**getattr(self, 'modelKwargs', {}))
        elif modelType == 'lasso':
            from sklearn.linear_model import Lasso

            return Lasso(**getattr(self, 'modelKwargs', {}))
        elif modelType == 'decisionTree':
            from sklearn.tree import DecisionTreeRegressor

            return DecisionTreeRegressor(**getattr(self, 'modelKwargs', {}))
        elif modelType == 'randomForest':
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(**getattr(self, 'modelKwargs', {}))
        elif modelType == 'pls':
            from sklearn.cross_decomposition import PLSRegression

            return PLSRegression(**getattr(self, 'modelKwargs', {}))

        elif modelType == 'pca_lr':
            from src.ml_tools import PCAPredictor
            from sklearn.linear_model import LinearRegression

            m = LinearRegression(fit_intercept=False)
            return PCAPredictor(m, **getattr(self, 'modelKwargs', {}))

        elif modelType == 'pca_ridge':
            from src.ml_tools import PCAPredictor
            from sklearn.linear_model import Ridge

            m = Ridge(fit_intercept=False)
            return PCAPredictor(m, **getattr(self, 'modelKwargs', {}))

        elif modelType == 'pca_lasso':
            from src.ml_tools import PCAPredictor
            from sklearn.linear_model import Lasso

            m = Lasso(fit_intercept=False)
            return PCAPredictor(m, **getattr(self, 'modelKwargs', {}))

        else:
            raise ValueError(
                'Unrecognized modelType {}. Should be one of [linearRegression, ridge, lasso, decisionTree, randomForest]'.format(
                    type(modelType)))

    def _initFromMolecules(self, m1, m2):
        template = None
        firstSample = None
        bestScore = 0

        for i in range(Chem.getNumConformations(m1)):
            Chem.applyConformation(m1, i)
            p = getPharmacophore(m1, fuzzy=self.fuzzy)
            self.aligner.addFeatures(p, True)

            for j in range(Chem.getNumConformations(m2)):
                Chem.applyConformation(m2, j)
                p2 = getPharmacophore(m2, fuzzy=self.fuzzy)
                self.aligner.addFeatures(p2, False)

                while self.aligner.nextAlignment():
                    tfMatrix = self.aligner.getTransform()
                    score = self.scorer(p, p2, tfMatrix)

                    if score is not None:
                        if score > bestScore:
                            template = p
                            Pharm.transform3DCoordinates(p2, tfMatrix)
                            firstSample = p2
                            bestScore = score

                self.aligner.clearEntities(False)
            self.aligner.clearEntities(True)

        # self.aligner.addFeatures(template, True)

        if template is None:
            raise AlignmentError

        # assign activities of first two molecules
        a1 = m1.getProperty(LOOKUPKEYS['activity'])
        a2 = m2.getProperty(LOOKUPKEYS['activity'])

        template.setProperty(LOOKUPKEYS['activity'], a1)
        for f in template:
            f.setProperty(LOOKUPKEYS['activity'], a1)

        firstSample.setProperty(LOOKUPKEYS['activity'], a2)
        for f in firstSample:
            f.setProperty(LOOKUPKEYS['activity'], a2)

        return template, firstSample, bestScore

    def log(self, p=None, message=None, name=None):
        if self.logPath is not None:
            if not os.path.isdir(self.logPath):
                os.makedirs(self.logPath)
            if message is not None:
                with open('{logPath}log.txt'.format(logPath=self.logPath), 'a') as f:
                    f.write(message + '\n')

            if p is not None:
                ID = str(p.getObjectID())
                savePharmacophore(p, '{logPath}{ID}.pml'.format(logPath=self.logPath, ID=ID if name is None else name))

    def fit(self, samples, **kwargs):
        """
        Makes an HP from the given collection of samples.
        :param samples:
        :param kwargs:
        :return:
        """
        if not isinstance(samples, Iterable):
            samples = [samples]

        self.onFitStart(samples, **kwargs)

        # align samples to template
        self._align(samples, **kwargs)

        # determine features
        self._determineFeatures(samples, **kwargs)

        # train ML model
        self._trainHPModel(samples, **kwargs)

        self.onFitEnd(samples, **kwargs)

    @abstractmethod
    def predict(self, samples, **kwargs):
        raise NotImplementedError

    def _align(self, samples, **kwargs):
        self.onAlignmentStart(samples, **kwargs)
        for s in samples:
            self.samples.append(s)
            try:
                alignedPharmacophore, score = self.align(s, returnScore=True, **kwargs)
                self.alignedSamples.append((alignedPharmacophore, score))
                self.nrSamples += 1
            except AlignmentError:
                pass
        self.onAlignmentEnd(samples, **kwargs)

    def _determineFeatures(self, samples, **kwargs):
        self.onFeatureDeterminationStart(samples, **kwargs)
        self.createHPFeatures(**kwargs)
        self.cleanFeatures(**kwargs)
        self.log(p=self.cleanedHP, name='cleanedHP_{}'.format(time.time()))
        self.onFeatureDeterminationEnd(samples, **kwargs)

    def _trainHPModel(self, samples, **kwargs):
        self.onMLTrainingStart(samples, **kwargs)
        self.trainHPModel(**kwargs)
        self.onMLTrainingEnd(samples, **kwargs)

    def align(self, sample, returnScore=False, **kwargs):
        """
        Align the given sample to the HP's model template. Keep in mind that the sample will not be aligned to the
        HP itself, but only its starting point. This has several  advantages:
        - speeds up training, since we only have to align molecules once and not again after changing some features
        - makes the alignment and HP creation process deterministic, since it no longer depends on the order of the
        samples as in the previous version (see pharmacophores_3.py).
        :param returnScore:
        :param sample:
        :param kwargs:
        :return:
        """
        if isinstance(sample, Chem.BasicMolecule):
            alignedPharmacophore, score = self._alignMolecule(sample, returnScore=returnScore, **kwargs)
        elif isinstance(sample, Pharm.BasicPharmacophore):
            alignedPharmacophore, score = self._alignPharmacophore(sample, returnScore=returnScore, **kwargs)
        else:
            raise TypeError('Given type {} not known. Type should be one of [Chem.BasicMolecule, Pharm.BasicPharmacophore].'.format(type(sample)))

        a = sample.getProperty(LOOKUPKEYS['activity'])
        alignedPharmacophore.setProperty(LOOKUPKEYS['activity'], a)
        for f in alignedPharmacophore:
            f.setProperty(LOOKUPKEYS['activity'], a)

        # create a new aligner / scorer object to delete references to any pharmacophores still in memory
        self.aligner = Pharm.PharmacophoreAlignment(True)
        self.scorer = Pharm.PharmacophoreFitScore()
        return alignedPharmacophore, score

    def _alignMolecule(self, mol, returnScore=True, returnAllConformations=False, **kwargs):
        """
        Align a molecule with multiple conformations (minimum 1) to the template. Returns the pharmacophore of the
        aligned molecule, as well as the alignment score if True.
        :param mol:
        :param returnScore:
        :param kwargs:
        :return:
        """
        bestScore = 0
        conformations = []
        bestPharmacophore = None
        self.aligner.addFeatures(self, True)

        for j in range(Chem.getNumConformations(mol)):
            Chem.applyConformation(mol, j)
            p = getPharmacophore(mol, fuzzy=self.fuzzy)
            self.aligner.addFeatures(p, False)

            while self.aligner.nextAlignment():
                tfMatrix = self.aligner.getTransform()
                score = self.scorer(self, p, tfMatrix)

                if score is not None:
                    Pharm.transform3DCoordinates(p, tfMatrix)
                    conformations.append((p, score))
                    if score > bestScore:
                        bestScore = score
                        bestPharmacophore = p
                else:
                    conformations.append((p, score))

            self.aligner.clearEntities(False)
        self.aligner.clearEntities(True)

        if bestScore == 0:
            raise AlignmentError

        if returnScore:
            if returnAllConformations:
                return bestPharmacophore, bestScore, conformations
            else:
                return bestPharmacophore, bestScore
        else:
            if returnAllConformations:
                return bestPharmacophore, conformations
            else:
                return bestPharmacophore

    def _alignPharmacophore(self, p, returnScore=True, **kwargs):
        """
        Align a given pharmacophore to the template. Aligns the aligned pharmacophore as well as th alignment score if
        True.
        :param p:
        :param returnScore:
        :param kwargs:
        :return:
        """
        if self.fuzzy:
            self.fuzzyfyPharmacophore(p, **kwargs)

        self.aligner.addFeatures(self, True)
        self.aligner.addFeatures(p, False)
        bestScore = 0
        while self.aligner.nextAlignment():
            tfMatrix = self.aligner.getTransform()
            score = self.scorer(self, p, tfMatrix)

            if score is not None:
                if score > bestScore:
                    Pharm.transform3DCoordinates(p, tfMatrix)
                    bestScore = score

        self.aligner.clearEntities(False)
        self.aligner.clearEntities(True)

        if bestScore == 0:
            raise AlignmentError

        if returnScore:
            return p, bestScore
        else:
            return p

    def fuzzyfyPharmacophore(self, p, **kwargs):
        for f in p:
            if Pharm.getType(f) == 5 or Pharm.getType(f) == 6:
                Pharm.clearOrientation(f)
                Pharm.setGeometry(f, Pharm.FeatureGeometry.SPHERE)

    @abstractmethod
    def createHPFeatures(self, **kwargs):
        """
        Creates the HP features from a cloud of individual pharmacophores. These features are obtained after aligning
        a collection of individual samples, but where the features are not merged during the alignment process, since
        this would lead to a non-deterministic outcome. Instead we align all samples beforehand and generate the HP
        features only afterwards from all the aligned pharmacophores.
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def trainHPModel(self, **kwargs):
        """
        Determines the feature activity and trains an ML model based on the features.

        Samples are accessible by self.samples, as well as aligned ones by self.alignedSamples.
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def cleanFeatures(self, **kwargs):
        """
        Creates a copy of the HP, containing all the features. Then sequentially removes features with ambiguous
        activity, which do not clearly determine the activity of the HP feature.

        Since we are no longer determining any feature activity, but only a method to obtained the features activity
        from the aligned query, I guess this method becomes obsolete. Even if there are features now with a high
        variance in the input, we fit a method to handle this variance. Therefore, all the features are of importance.

        If overwriting this method, make sure to call this method of the parent class or define the self.cleanedHP
        object yourself, since it will be relevant for subsequent functions.
        :param kwargs:
        :return:
        """
        pass

    def onFeatureDeterminationStart(self, samples, **kwargs):
        pass

    def onFeatureDeterminationEnd(self, samples, **kwargs):
        pass

    def onMLTrainingStart(self, samples, **kwargs):
        pass

    def onMLTrainingEnd(self, samples, **kwargs):
        pass

    def onAlignmentStart(self, samples, **kwargs):
        pass

    def onAlignmentEnd(self, samples, **kwargs):
        pass

    def onFitStart(self, samples, **kwargs):
        pass

    def onFitEnd(self, samples, **kwargs):
        pass

    def load(self, path, **kwargs):
        raise NotImplementedError

    def save(self, path, **kwargs):
        raise NotImplementedError


class DistanceHyperpharmacophore(HyperPharmacophore):

    """
    This is basically just a reimplementation of the 'classical' Hyperpharmacophore from a previous version. Instead
    of using and determining feature activities, the distance to the corresponding feature is put into the feature
    matrix, since in the previous version the feature activity had always the same values, therefore acting only as
    a binary indicator whether the feature was present or not and variability only coming from the distance to the
    respective feature.
    Distance can be either typical inverse of distance or gaussian based, where the value decreases exponentially
    with increasing distance.
    """

    def __init__(self,
                 template=None,
                 distanceType='inverse',
                 maxDistance=1e3,
                 weightType='distance',
                 addFeatureFrequencyWeight=False,
                 **kwargs
                 ):
        super(DistanceHyperpharmacophore, self).__init__(template, weightType=weightType, **kwargs)
        self.addFeatureFrequencyWeight = addFeatureFrequencyWeight
        self.distanceType = distanceType
        self.maxDistance = maxDistance
        self.trainingDim = None
        self.gaussianFeatures = []  # same order as features in self

    def onFeatureDeterminationEnd(self, samples, **kwargs):
        if self.distanceType == 'gaussian':  # create gaussian features so we can calculate distance easy
            raise NotImplementedError
            # from ShapeAlignment import prepareForShapeAlignment
            #
            # for f in self.cleanedHP:
            #     featureSet = Pharm.FeatureSet()
            #     featureSet.addFeature(f)
            #     featureShape, featureShapeFunction = prepareForShapeAlignment(featureSet)
            #     self.gaussianFeatures.append((featureShape, featureShapeFunction))

    def trainHPModel(self, aggregateEnvironment=False, **kwargs):
        """
        Extract the feature data from the samples and train one or multiple ML models on this data.
        :param aggregateEnvironment: Indicates whether all features of the same type are taken into account or just
        the single closest feature.
        :param kwargs:
        :return:
        """
        # extract feature data
        featureData = []
        y_true = []
        for sample, score in self.alignedSamples:
            y_true.append(sample.getProperty(LOOKUPKEYS['activity']))
            features = self.getFeatureData(sample, aggregateEnvironment=aggregateEnvironment, **kwargs)
            featureData.append(features)

        if len(featureData) == 0:
            return

        featureData = np.stack(featureData, axis=0)
        y_true = np.array(y_true)

        if featureData.shape[1] == 0:
            self.trainingDim = 0
            return
        else:
            self.trainingDim = featureData.shape[1]

        if self.modelType == 'pls' and (featureData.shape[0] > self.mlModel.n_components or featureData.shape[1] > self.mlModel.n_components):
            self.trainingDim = 0  # not really 0, but predict checks for 0 if something went wrong
            return

        # fit ml model
        if isinstance(self.mlModel, Iterable) and self.modelType != 'randomForest':
            for m in self.mlModel:
                m.fit(featureData, y_true)
        else:
            self.mlModel.fit(featureData, y_true)

    def getFeatureData(self, p, aggregateEnvironment=False, **kwargs):
        features = []
        for f in self.cleanedHP:
            # get contribution from corresponding feature(s) in query
            if aggregateEnvironment:
                # get all features of the same type and aggregate their contribution based on the distance
                raise NotImplementedError
            else:
                # get closest feature
                reference, distance = getClosestFeature(f, p, **kwargs)  # reference is feature of query pharmacophore
                if reference is None:
                    features.append(0)  # set weight to zero
                    continue

                # get weight from distance
                if self.weightType == 'gaussian':
                    index = self.cleanedHP.getFeatureIndex(f)  # get feature of HP
                    gaussianFeature = self.gaussianFeatures[index]
                    weight = getGaussianWeight(reference, gaussianFeature, **kwargs)

                elif self.weightType == 'distance':
                    weight = getDistanceWeight(distance, maxDistance=self.maxDistance, **kwargs)

                elif self.weightType == 'nrOfFeatures':
                    weight = getFeatureFrequencyWeight(f, len(self.alignedSamples), LOOKUPKEYS, **kwargs)

                elif self.weightType is None or self.weightType == 'binary':
                    weight = 1

                else:
                    raise TypeError('weightType not recognized')

                if self.addFeatureFrequencyWeight:
                    weight = weight*getFeatureFrequencyWeight(f, len(self.alignedSamples), LOOKUPKEYS, **kwargs)

                features.append(weight)
        return features

    def createHPFeatures(self, **kwargs):
        featureSet = Pharm.BasicPharmacophore()
        for s, score in self.alignedSamples:
            for f in s:
                added = featureSet.addFeature()
                added.assign(f)
        self.log(p=featureSet, name='featureSet_{}'.format(time.time()))
        clusters = self.clusterFeatures(featureSet, **kwargs)

        # now that we have the clusters, we need to place features, which represent the clusters. More than one feature
        # can be placed to represent clusters. The important part is that ALL the features in the feature set are
        # finally represented by a single HP feature, whereas a single HP feature can represent more than one feature.
        def createNewFeature(coordinates, featureType, **kwargs):
            addedFeature = self.cleanedHP.addFeature()
            Pharm.setType(addedFeature, featureType)
            if not isinstance(coordinates, Math.Vector3D):
                c = Math.Vector3D()
                for i, value in enumerate(coordinates):
                    c.setElement(i, float(value))
                coordinates = c
            Chem.set3DCoordinates(addedFeature, coordinates)
            Pharm.setGeometry(addedFeature, Pharm.FeatureGeometry.SPHERE)
            Pharm.setTolerance(addedFeature, 1.5)  # set default tolerance

            return addedFeature

        def placeOneFeature(cluster, featureType, threshold=None, **kwargs):
            """
            Place a feature in the 'center of gravity' of all features.
            If it does not work, try the center.
            If this does not work, check whether one feature can reach all the others.
            :return:
            """
            success = False

            allCoordinates = []
            for i in cluster:
                f = featureSet.getFeature(i)
                allCoordinates.append(Chem.get3DCoordinates(f).toArray())
            allCoordinates = np.stack(allCoordinates, axis=0)

            # calculate center
            pass

            # calculate center of gravity
            center = np.mean(allCoordinates, axis=0)
            canReachAll = True
            for i in cluster:
                f = featureSet.getFeature(i)
                r = Pharm.getTolerance(f)
                c = Chem.get3DCoordinates(f).toArray()
                if threshold is None:
                    t = self.threshold
                else:
                    t = threshold
                if calculateDistance(center, c) > t:  # center can not reach all features -> break
                    canReachAll = False
                    break

            if canReachAll:
                success = True
                addedFeature = createNewFeature(center, featureType)
                activities = []  # store activities of all merged features
                for i in cluster:
                    f = featureSet.getFeature(i)
                    activities.append(f.getProperty(LOOKUPKEYS['activity']))
                addedFeature.setProperty(LOOKUPKEYS['activities'], activities)
                addedFeature.setProperty(LOOKUPKEYS['nrOfFeatures'], len(cluster))
                return success

            # check whether one matches all
            for i in range(len(cluster)):
                canReachAll = True
                f1 = featureSet.getFeature(cluster[i])
                r1 = Pharm.getTolerance(f1)
                c1 = Chem.get3DCoordinates(f1).toArray()
                for j in range(len(cluster)):
                    if i == j:
                        continue

                    f2 = featureSet.getFeature(cluster[j])
                    r2 = Pharm.getTolerance(f2)
                    c2 = Chem.get3DCoordinates(f2).toArray()
                    if threshold is None:
                        t = self.threshold
                    else:
                        t = threshold
                    if calculateDistance(c1, c2) > t:  # feature i can not reach all features -> break inner loop
                        canReachAll = False
                        break

                if canReachAll:
                    addedFeature = self.cleanedHP.addFeature()
                    addedFeature.assign(f1)
                    activities = []  # store activities of merged features
                    for j in range(len(cluster)):
                        if i == j:
                            continue

                        f2 = featureSet.getFeature(cluster[j])
                        activities.append(f2.getProperty(LOOKUPKEYS['activity']))
                    addedFeature.setProperty(LOOKUPKEYS['activities'], activities)
                    addedFeature.setProperty(LOOKUPKEYS['nrOfFeatures'], len(cluster))
                    success = True
                    return success

            return success

        def placeMultipleFeatures(cluster, featureType, threshold=None, **kwargs):
            """
            Start by placing a feature on the 'most' connected feature. Then go on like this by chosing features
            in decreasing order of 'connectedness' until all features are consumed.
            :return:
            """
            # get connectedness to corresponding features
            features = []
            connectedness = []
            neighbors = []
            for i in range(len(cluster)):
                nrNeighbors = 0
                tempNeighbors = []
                f1 = featureSet.getFeature(cluster[i])
                r1 = Pharm.getTolerance(f1)
                c1 = Chem.get3DCoordinates(f1)
                features.append(f1)

                for j in range(len(cluster)):
                    if i == j:
                        continue

                    f2 = featureSet.getFeature(cluster[j])
                    r2 = Pharm.getTolerance(f2)
                    c2 = Chem.get3DCoordinates(f2)

                    if threshold is None:
                        t = self.threshold
                    else:
                        t = threshold
                    if calculateDistance(c1, c2) < t:
                        nrNeighbors += 1
                        tempNeighbors.append(j)

                connectedness.append(nrNeighbors)
                neighbors.append(tempNeighbors)

            # sort by connectedness
            sortedArgs = np.argsort(connectedness)  # sorts in ascending order

            # iterate in descending order over features and merge them with neighbors until no features are left
            mergeOnce = kwargs.get('mergeOnce', True)
            if mergeOnce:
                alreadyMerged = set()
                for i in reversed(sortedArgs):
                    if i in alreadyMerged:
                        continue

                    alreadyMerged.add(i)
                    f = features[i]
                    addedFeature = self.cleanedHP.addFeature()
                    addedFeature.assign(f)
                    activities = addedFeature.getProperty(LOOKUPKEYS['activities'])
                    nrOfMerged = 0
                    for n in neighbors[i]:  # iterate over neighbor indices
                        if n in alreadyMerged:
                            continue

                        alreadyMerged.add(n)
                        f2 = features[n]
                        activities.append(f2.getProperty(LOOKUPKEYS['activity']))
                        nrOfMerged += 1

                    addedFeature.setProperty(LOOKUPKEYS['activities'], activities)
                    addedFeature.setProperty(LOOKUPKEYS['nrOfFeatures'], nrOfMerged)

            else:
                for i in reversed(sortedArgs):
                    f = features[i]
                    addedFeature = self.cleanedHP.addFeature()
                    addedFeature.assign(f)
                    activities = addedFeature.getProperty(LOOKUPKEYS['activities'])
                    for n in neighbors[i]:
                        f2 = features[n]
                        activities.append(f2.getProperty(LOOKUPKEYS['activity']))

                    addedFeature.setProperty(LOOKUPKEYS['nrOfFeatures'], len(neighbors))

            return

        # process clusters of each feature type
        for ft, featureClusters in clusters.items():
            for cluster in featureClusters:
                if len(cluster) == 1:  # easy. Only one feature which can immediately be added
                    f = featureSet.getFeature(cluster[0])
                    added = self.cleanedHP.addFeature()
                    added.assign(f)
                    added.setProperty(LOOKUPKEYS['activities'], [added.getProperty(LOOKUPKEYS['activity'])])
                    added.setProperty(LOOKUPKEYS['nrOfFeatures'], 1)

                else:  # multiple features are in this cluster.
                    # One or more features need to be found, which can represent the features in the cluster
                    # Features are added to the cleanedHP inside the functions
                    success = placeOneFeature(cluster, ft)  # try placing on feature to match all
                    if not success:
                        placeMultipleFeatures(cluster, ft)

    def clusterFeatures(self, featureSet, threshold=None, **kwargs):
        """
        Clusters the features of the aligned pharmacophores by a hierarchical clustering algorithm.
        Basically, all features within a certain distance are put into a single cluster.

        If the treshold is not given, per default the radius of the pharmacophore features will be used.

        :param featureSet:
        :param threshold:
        :param kwargs:
        :return:
        """
        from src.pharmacophore_tools import FEATURE_TYPES
        # cluster the features and store a list of clusters, whereas each feature is referenced in the list by its index
        clusters = {ft: [] for ft in FEATURE_TYPES.values()}
        alreadyInCluster = set()

        for i in range(featureSet.numFeatures):  # TODO: use clustering algorithm with max-cutoff
            if i in alreadyInCluster:
                continue

            alreadyInCluster.add(i)
            f1 = featureSet.getFeature(i)
            c1 = Chem.get3DCoordinates(f1).toArray()
            temp = [i]  # holds indices of current cluster
            featureType = Pharm.getType(f1)

            for j in range(i+1, featureSet.numFeatures):
                if j in alreadyInCluster:
                    continue

                f2 = featureSet.getFeature(j)
                if Pharm.getType(f2) != featureType:
                    continue

                c2 = Chem.get3DCoordinates(f2).toArray()
                if threshold is None:
                    t = self.threshold
                else:
                    t = threshold
                if calculateDistance(c1, c2) < t:
                    temp.append(j)
                    alreadyInCluster.add(j)

            clusters[featureType].append(temp)
        return clusters

    def cleanFeatures(self, **kwargs):
        """
        Remove ambiguous features.
        Each feature should at least have two entries --> one merged feature.
        The range of activity values for each feature should not be more than half the entire range of feature activity
        of all features.
        :param kwargs:
        :return:
        """
        super(DistanceHyperpharmacophore, self).cleanFeatures(**kwargs)

        # determine min and max value of activity.
        toRemove = []
        startingActivities = self.cleanedHP.getFeature(0).getProperty(LOOKUPKEYS['activities'])
        refMin, refMax = min(startingActivities), max(startingActivities)
        storedActivities = []
        for i in range(self.cleanedHP.numFeatures):
            activities = self.cleanedHP.getFeature(i).getProperty(LOOKUPKEYS['activities'])

            if min(activities) < refMin:
                refMin = min(activities)
            if max(activities) > refMax:
                refMax = max(activities)

            if len(activities) == 1:
                toRemove.append(i)  # remove features with only one feature activity
                continue
            else:
                storedActivities.append(activities)  # store activities, so we do not have to iterate the features again

        for i in reversed(toRemove):
            self.cleanedHP.removeFeature(i)

        # query the features for their range
        toRemove = []
        for i, activities in enumerate(storedActivities):
            minValue, maxValue = min(activities), max(activities)
            if maxValue-minValue > (refMax-refMin)/2:
                toRemove.append(i)
        for i in reversed(toRemove):
            self.cleanedHP.removeFeature(i)

    def predict(self, samples, aggregateEnvironment=False, returnScores=False, **kwargs):
        if self.trainingDim == 0:
            if returnScores:
                return np.array([0] * len(samples)), [0] * len(samples)
            else:
                return np.array([0] * len(samples))

        if not isinstance(samples, Iterable):
            samples = [samples]

        # get feature data from prediction samples
        scores, featureData = [], []
        for s in samples:
            try:
                alignedPharmacophore, score = self.align(s, returnScore=True, **kwargs)
                features = self.getFeatureData(alignedPharmacophore, aggregateEnvironment=aggregateEnvironment, **kwargs)
            except AlignmentError:
                score = 0
                features = [0]*self.cleanedHP.numFeatures

            featureData.append(features)
            scores.append(score)

        # predict activity for samples
        featureData = np.stack(featureData, axis=0)
        if isinstance(self.mlModel, Iterable) and self.modelType != 'randomForest':
            y_pred = []
            for m in self.mlModel:
                predictions = m.predict(featureData)
                y_pred.append(predictions)
            y_pred = np.mean(np.stack(y_pred, axis=0), axis=0).flatten()  # weighed average

        else:
            y_pred = self.mlModel.predict(featureData).flatten()

        # handle unaligned samples -> set predictions to zero
        y_pred = np.where((scores == 0), np.zeros(len(scores)), y_pred)

        if returnScores:
            return y_pred, scores
        else:
            return y_pred

    def save(self, path, **kwargs):
        """
        Saves everything needed to a folder to restore the model. This includes:
        - the initial template to which we align new samples
        - all aligned samples
        - the HP model after clustering the features
        - the ml model to predict new samples
        :param path:
        :param kwargs:
        :return:
        """
        import pickle
        
        if not os.path.isdir(path):
            os.makedirs(path)

        savePharmacophore(self, '{}template.pml'.format(path))
        if not os.path.isdir('{}alignedSamples'.format(path)):
            os.mkdir('{}alignedSamples'.format(path))
        for i, alignedSample in enumerate(self.alignedSamples): 
            savePharmacophore(alignedSample[0], '{}alignedSamples/{}.pml'.format(path, str(i)))
        savePharmacophore(self.cleanedHP, '{}hpModel.pml'.format(path))
        if isinstance(self.mlModel, Iterable) and self.modelType != 'randomForest': 
            os.mkdir('{}mlModel/'.format(path))
            for i, m in enumerate(self.mlModel):
                with open('{}mlModel/model_{}.pkl'.format(path, str(i)), 'wb') as mlfile:
                    pickle.dump(m, mlfile)
        else:
            with open('{}mlModel.pkl'.format(path), 'wb') as mlfile:
                pickle.dump(self.mlModel, mlfile)

        features = {i: {name: self.cleanedHP.getFeature(i).getProperty(key) for name, key in LOOKUPKEYS.items() if name != 'prediction'} for i in range(self.cleanedHP.numFeatures)}
        with open('{}featureProperties.json'.format(path), 'w') as f:
            json.dump(features, f)

        parameters = {
            'fuzzy': self.fuzzy,
            'weightType': self.weightType,
            'logPath': self.logPath,
            'name': self.name,
            'maxDistance': self.maxDistance,
            'nrSamples': self.nrSamples,
            'distanceType': self.distanceType,
            'trainingDim': self.trainingDim,
            'modelType': self.modelType,
            'modelKwargs': self.modelKwargs,
            'threshold': self.threshold,
        }
        with open('{}parameters.json'.format(path), 'w') as f:
            json.dump(parameters, f, indent=2)

    def load(self, path, **kwargs):
        import pickle

        with open('{}parameters.json'.format(path), 'r') as f:
            parameters = json.load(f)
        for key, value in parameters.items():
            setattr(self, key, value)
        
        print('Loading HP model from {}'.format(path))
        
        # load template
        if os.path.isfile('{}template.pml'.format(path)):
            self.template = loadPharmacophore('{}template.pml'.format(path))
            self.assign(self.template)
        else:
            print('Could not find template at file: template.pml')
            
        # load aligned samples
        if not os.path.isdir('{}alignedSamples'.format(path)): 
            print('Could not find folder for aligned samples')
        else:
            alignedSamples = []
            for f in os.listdir('{}alignedSamples'.format(path)):
                s = loadPharmacophore('{}alignedSamples/{}'.format(path, f))
                alignedSamples.append(s)
            self.alignedSamples = alignedSamples
        
        # load hp model
        if os.path.isfile('{}hpModel.pml'.format(path)): 
            self.cleanedHP = loadPharmacophore('{}hpModel.pml'.format(path))
            if os.path.isfile('{}featureProperties.json'.format(path)):
                with open('{}featureProperties.json'.format(path), 'r') as f:
                    features = json.load(f)
                for i, properties in features.items():
                    for name, p in properties.items():
                        self.cleanedHP.getFeature(int(i)).setProperty(LOOKUPKEYS[name], p)
        else:
            print('Could not load hp model at file: hpModel.pml')
            
        # load ml model
        if os.path.isdir('{}mlModel'.format(path)):
            mlModel = []
            for f in os.listdir('{}mlModel'.format(path)):
                with open('{}mlModel/{}'.format(path, f), 'rb') as mlFile:
                    mlModel.append(pickle.load(mlFile))
            self.mlModel = mlModel
        elif os.path.isfile('{}mlModel.pkl'.format(path)):
            with open('{}mlModel.pkl'.format(path, f), 'rb') as mlFile:
                self.mlModel = pickle.load(mlFile)
        else:
            print('Could not find ml model at file: mlModel.pkl or folder: mlModel/')
