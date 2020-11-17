import CDPL.Pharm as Pharm
import CDPL.Chem as Chem
import CDPL.Base as Base
import CDPL.Math as Math
from random import shuffle
import os
import json
import time
import numpy as np
from collections.abc import Iterable
from abc import abstractmethod
import signal
from Hyperpharmacophore.utils import runTimeHandler, AlignmentError, getClosestFeature, getDistanceWeight, getGaussianWeight, calculateDistance, getFeatureFrequencyWeight
from Pharmacophore_tools import get_pharmacophore, save_pharmacophore


signal.signal(signal.SIGALRM, runTimeHandler)


LOOKUPKEYS = {
    'activity': Base.LookupKey.create('activity'),
    'activities': Base.LookupKey.create('activities'),
    'nrOfFeatures': Base.LookupKey.create('nrOfFeatures'),
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

            return LinearRegression(**modelKwargs)
        elif modelType == 'ridge':
            from sklearn.linear_model import Ridge

            return Ridge(**modelKwargs)
        elif modelType == 'lasso':
            from sklearn.linear_model import Lasso

            return Lasso(**modelKwargs)
        elif modelType == 'decisionTree':
            from sklearn.tree import DecisionTreeRegressor

            return DecisionTreeRegressor(**modelKwargs)
        elif modelType == 'randomForest':
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(**modelKwargs)
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
            p = get_pharmacophore(m1, fuzzy=self.fuzzy)
            self.aligner.addFeatures(p, True)

            for j in range(Chem.getNumConformations(m2)):
                Chem.applyConformation(m2, j)
                p2 = get_pharmacophore(m2, fuzzy=self.fuzzy)
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
                save_pharmacophore(p, '{logPath}{ID}.pml'.format(logPath=self.logPath, ID=ID if name is None else name))

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
            p = get_pharmacophore(mol, fuzzy=self.fuzzy)
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
        # for f in self:
        #     newFeature = self.cleanedHP.addFeature()
        #     newFeature.assign(f)
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
            from ShapeAlignment import prepareForShapeAlignment

            for f in self.cleanedHP:
                featureSet = Pharm.FeatureSet()
                featureSet.addFeature(f)
                featureShape, featureShapeFunction = prepareForShapeAlignment(featureSet)
                self.gaussianFeatures.append((featureShape, featureShapeFunction))

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

                elif self.weightTyoe == 'nrOfFeatures':
                    weight = getFeatureFrequencyWeight(f, len(self.alignedSamples), LOOKUPKEYS, **kwargs)

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
                    t = r
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
                        t = max(r1, r2)
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
                    # addedFeature.setProperty(LOOKUPKEYS['activity'], f1.getProperty(LOOKUPKEYS['activity']))
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
                        t = max(r1, r2)
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
                    # addedFeature.setProperty(LOOKUPKEYS['activity'], f.getProperty(LOOKUPKEYS['activity']))
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

                    # addedFeature.setProperty(LOOKUPKEYS['activity'], f.getProperty(LOOKUPKEYS['activity']))
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
                    # added.setProperty(LOOKUPKEYS['activity'], added.getProperty(LOOKUPKEYS['activity']))
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
        from Pharmacophore_tools import FEATURE_TYPES
        # cluster the features and store a list of clusters, whereas each feature is referenced in the list by its index
        clusters = {ft: [] for ft in FEATURE_TYPES.values()}
        alreadyInCluster = set()

        for i in range(featureSet.numFeatures):
            if i in alreadyInCluster:
                continue

            alreadyInCluster.add(i)
            f1 = featureSet.getFeature(i)
            r1 = Pharm.getTolerance(f1)
            c1 = Chem.get3DCoordinates(f1).toArray()
            temp = [i]  # holds indices of current cluster
            featureType = Pharm.getType(f1)

            for j in range(i+1, featureSet.numFeatures):
                if j in alreadyInCluster:
                    continue

                f2 = featureSet.getFeature(j)
                if Pharm.getType(f2) != featureType:
                    continue

                r2 = Pharm.getTolerance(f2)
                c2 = Chem.get3DCoordinates(f2).toArray()
                if threshold is None:
                    t = max(r1, r2)
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
            if len(activities) == 1:
                toRemove.append(i)  # remove features with only one feature activity
                continue
            else:
                storedActivities.append(activities)  # store activities, so we do not have to iterate the features again

            if min(activities) < refMin:
                refMin = min(activities)
            if max(activities) > refMax:
                refMax = max(activities)

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
            return [0] * len(samples), [0] * len(samples)

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
            y_pred = np.sum(np.stack(y_pred, axis=0), axis=0).flatten()

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

        save_pharmacophore(self, '{}template.pml'.format(path))
        if not os.path.isdir('{}alignedSamples'.format(path)):
            os.mkdir('{}alignedSamples'.format(path))
        for i, alignedSample in enumerate(self.alignedSamples): 
            save_pharmacophore(alignedSample[0], '{}alignedSamples/{}.pml'.format(path, str(i)))
        save_pharmacophore(self.cleanedHP, '{}hpModel.pml'.format(path))
        if isinstance(self.mlModel, Iterable) and self.modelType != 'randomForest': 
            os.mkdir('{}mlModel/'.format(path))
            for i, m in enumerate(self.mlModel):
                with open('{}mlModel/model_{}.pkl'.format(path, str(i)), 'wb') as mlfile:
                    pickle.dump(m, mlfile)
        else:
            with open('{}mlModel.pkl'.format(path), 'wb') as mlfile:
                pickle.dump(self.mlModel, mlfile)

        parameters = {
            'fuzzy': self.fuzzy,
            'weightType': self.weightType,
            'logPath': self.logPath,
            'name': self.name,
            'maxDistance': self.maxDistance,
            'nrSamples': self.nrSamples,
            'distanceType': self.distanceType,
            'trainingDim': self.trainingDim,
        }
        with open('{}parameters.json'.format(path), 'w') as f:
            json.dump(parameters, f, indent=2)

    def load(self, path, **kwargs):
        import pickle
        from Pharmacophore_tools import load_pml_pharmacophore

        with open('{}parameters.json'.format(path), 'r') as f:
            parameters = json.load(f)
        for key, value in parameters.items():
            setattr(self, key, value)
        
        print('Loading HP model from {}'.format(path))
        
        # load template
        if os.path.isfile('{}template.pml'.format(path)):
            self.template = load_pml_pharmacophore('{}template.pml'.format(path))
        else:
            print('Could not find template at file: template.pml')
            
        # load aligned samples
        if not os.path.isdir('{}alignedSamples'.format(path)): 
            print('Could not find folder for aligned samples')
        else:
            alignedSamples = []
            for f in os.listdir('{}alignedSamples'.format(path)):
                s = load_pml_pharmacophore('{}alignedSamples/{}'.format(path, f))
                alignedSamples.append(s)
            self.alignedSamples = alignedSamples
        
        # load hp model
        if os.path.isfile('{}hpModel.pml'.format(path)): 
            self.cleanedHP = load_pml_pharmacophore('{}hpModel.pml'.format(path))
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
                

class GradientHyperpharmacohpore(HyperPharmacophore):

    """
    This class implements a gradient based way to learn to the activity values of the features. Essentially, what is
    learned is not a fixed activity value for each feature, but only a model which estimates the features activity based
    on the input. Each feature represents a simple ML Model, which is jointly trained with a final model. The final
    model takes into account the outputs of all the intermediate models, to generate a final activity estimate for the
    pharmacophore. Since no endpoint exists for the intermediate models (we have no clue which of the features is more
    important), an error is calculated at the end. Based on this error we can calculate a gradient which is being used
    to update the parameters of the intermediate models. Then again, the final model is fitted on the intermediate
    predictions and an error is calculated for the predicted activity. This process is repeated a few times, until
    either it converges or the performance of the validation set decreases.
    """

    def __init__(self, **kwargs):
        super(GradientHyperpharmacohpore, self).__init__(**kwargs)
        self.kwargs = kwargs


class HMMlikeHyperpharmacophore(HyperPharmacophore):

    """
    This class implements an HMM like algorithm to determine the features "activity". Based on the different activities
    associated with a feature, we try to estimate the underlying distribution yielding these activity values.
    At inference, the distribution is evaluated at the respective location yielding an activity value for the feature
    which is being used in a simple ML model to determine the activity of the pharmacophore.
    """

    def __init__(self, **kwargs):
        super(HMMlikeHyperpharmacophore, self).__init__(**kwargs)
        self.kwargs = kwargs


class SequentialHyperpharmacophore(Pharm.BasicPharmacophore):

    """
    A reimplementation of the 'old' HP, with the 'old' alignment and creation procedure.
    First a template is given, or obtained from two molecules. Then the samples are sequentially aligned to the HP,
    whereas after each alignment the features are merged or added to the HP. In contrast to the Hyperpharmacophore
    defined in this script, the final results are dependent on the order of alignment of individual samples.

    However, the problem of clustering and creating features afterwards is avoided.
    """

    def __init__(self, template,
                 modelType='linearRegression',
                 weightType='distance',
                 modelKwargs=None,
                 logPath=None,
                 name='Hyperpharmacophore',
                 alignmentTimeout=30,
                 fuzzy=True,
                 maxDistance=1e3,
                 **kwargs):
        super(SequentialHyperpharmacophore, self).__init__()
        self.kwargs = kwargs

        self.template = template
        self.modelType = modelType
        self.weightType = weightType
        self.modelKwargs = {} if modelKwargs is None else modelKwargs
        self.logPath = logPath
        self.name = name
        self.alignmentTimeout = alignmentTimeout
        self.fuzzy = fuzzy
        self.maxDistance = maxDistance

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
            raise TypeError(
                'Unrecognized modelType {}. Should be either a str or a list strings'.format(type(self.modelType)))

        # some collections to store data
        self.samples = []  # contains initial samples, without any specific conformation
        self.alignedSamples = []  # contains tuples of aligned pharmacophores and scores
        self.cleanedHP = Pharm.BasicPharmacophore()
        self.nrSamples = 0
        self.trainingDim = None
        self.varianceMask = None
        self.correlationMask = None

        # init template
        if isinstance(template, Pharm.BasicPharmacophore):
            self.nrSamples += 1
            a = template.getProperty(LOOKUPKEYS['activity'])
            for f in template:
                self.initHPFeature(f, activity=a, activities=[a])

            self.samples.append(template)
            self.alignedSamples.append((template, 0))

        elif isinstance(template, Iterable) and len(template) == 2:  # assume molecules
            m1, m2 = template[0], template[1]
            if not isinstance(m1, Chem.BasicMolecule) or not isinstance(m2, Chem.BasicMolecule):
                raise TypeError(
                    'If template is iterable, length needs to be 2, whereas both entities need to be of type Chem.BasicMolecule')

            template, firstSample, score = self._initFromMolecules(m1, m2)
            a = template.getProperty(LOOKUPKEYS['activity'])
            for f in template:
                self.initHPFeature(f, activity=a, activities=[a])
            self.nrSamples += 2
            self.samples.extend([m1, m2])
            self.alignedSamples.extend([(template, score), (firstSample, score)])
            self.merge(firstSample, **kwargs)

        elif template is None:  # only relevant for subclass OrthogonalHP
            pass
        else:
            raise TypeError(
                'Given type {} not known. Type should be one of [Chem.BasicMolecule, Pharm.BasicPharmacophore].'.format(
                    type(template)))

    def initHPFeature(self, f, activity=None, activities=None, nrOfFeatures=1):
        newFeature = self.addFeature()
        newFeature.assign(f)
        if activity is not None:
            newFeature.setProperty(LOOKUPKEYS['activity'], activity)
        if activities is not None:
            newFeature.setProperty(LOOKUPKEYS['activities'], activities)
        newFeature.setProperty(LOOKUPKEYS['nrOfFeatures'], nrOfFeatures)

    def mergeToHPFeature(self, feature, hpFeature, activity=None, **kwargs):
        # merge activities
        activities = hpFeature.getProperty(LOOKUPKEYS['activities'])
        activities.append(activity)  # is view on list -> don't need to set specifically
        nrFeatures = hpFeature.getProperty(LOOKUPKEYS['nrOfFeatures'])
        hpFeature.setProperty(LOOKUPKEYS['nrOfFeatures'], nrFeatures+1)

        # merge coordinates
        refC = Chem.get3DCoordinates(hpFeature)
        queryC = Chem.get3DCoordinates(feature)
        newC = (nrFeatures * refC.toArray() + queryC.toArray()) / (nrFeatures + 1)
        for i in range(len(newC)):
            refC[i] = newC[i]
        Chem.set3DCoordinates(hpFeature, newC)

    def merge(self, pharmacophore, **kwargs):
        a = pharmacophore.getProperty(LOOKUPKEYS['activity'])

        # determine correpsonding features
        matchingFeatures, nonMatchingFeatures = [], []
        for f in pharmacophore:
            ref, distance = getClosestFeature(f, self, **kwargs)
            if ref is not None:
                matchingFeatures.append((f, ref))
            else:
                nonMatchingFeatures.append(f)

        # merge corresponding features
        for f, hpf in matchingFeatures:
            self.mergeToHPFeature(f, hpf, activity=a, **kwargs)

        # init new features which have no corresponding feature in HP
        for f in nonMatchingFeatures:
            self.initHPFeature(f, activity=a, activities=[a])

    def _initMLModel(self, modelType, modelKwargs):

        # set up ml model
        if modelType == 'linearRegression':
            from sklearn.linear_model import LinearRegression

            return LinearRegression(**modelKwargs)
        elif modelType == 'ridge':
            from sklearn.linear_model import Ridge

            return Ridge(**modelKwargs)
        elif modelType == 'lasso':
            from sklearn.linear_model import Lasso

            return Lasso(**modelKwargs)
        elif modelType == 'decisionTree':
            from sklearn.tree import DecisionTreeRegressor

            return DecisionTreeRegressor(**modelKwargs)
        elif modelType == 'randomForest':
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(**modelKwargs)
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
            p = get_pharmacophore(m1, fuzzy=self.fuzzy)
            self.aligner.addFeatures(p, True)

            for j in range(Chem.getNumConformations(m2)):
                Chem.applyConformation(m2, j)
                p2 = get_pharmacophore(m2, fuzzy=self.fuzzy)
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

        if template is None:
            raise AlignmentError

        # assign activities of first two molecules
        a1 = m1.getProperty(LOOKUPKEYS['activity'])
        a2 = m2.getProperty(LOOKUPKEYS['activity'])

        template.setProperty(LOOKUPKEYS['activity'], a1)
        for f in template:
            f.setProperty(LOOKUPKEYS['activity'], a1)
            # f.setProperty(LOOKUPKEYS['nrOfFeatures'], 1)

        firstSample.setProperty(LOOKUPKEYS['activity'], a2)
        for f in firstSample:
            f.setProperty(LOOKUPKEYS['activity'], a2)
            # f.setProperty(LOOKUPKEYS['nrOfFeatures'], 1)

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
                save_pharmacophore(p, '{logPath}{ID}.pml'.format(logPath=self.logPath,
                                                                 ID=ID if name is None else name))

    def fit(self, samples, **kwargs):
        """
        Makes an HP from the given collection of samples.
        :param samples:
        :param kwargs:
        :return:
        """
        if not isinstance(samples, Iterable):
            samples = [samples]

        samples = self.onFitStart(samples, **kwargs)

        # align samples to template
        self._align(samples, **kwargs)

        # train ML model
        self._trainHPModel(samples, **kwargs)

        self.onFitEnd(samples, **kwargs)

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
        # alignedSamples = []

        for sample in self.samples:  # align samples again, since model has likely changed since last alignment
            try:
                returnDict = self.align(sample, returnScore=True, **kwargs)  # TODO: maybe align to cleaned HP instead of self -> pass reference keyword as kwarg to align fn
                alignedPharmacophore, score = returnDict['alignedPharmacophore'], returnDict['score']
                # alignedSamples.append((alignedPharmacophore, score))
                y_true.append(alignedPharmacophore.getProperty(LOOKUPKEYS['activity']))
                features = self.getFeatureData(alignedPharmacophore, aggregateEnvironment=aggregateEnvironment, **kwargs)
                featureData.append(features)

            except AlignmentError:
                pass

        if len(featureData) == 0:
            return

        featureData = np.stack(featureData, axis=0)
        y_true = np.array(y_true)

        # post-process featureData for low Variance or highly correlated columns
        processed = self.postProcessFeatureData(featureData, training=True, **kwargs)

        if processed.shape[1] == 0:
            print('Could not fit model on trainings data, since no features were left after post-processing.')
            self.trainingDim = 0
            return
        else:
            self.trainingDim = processed.shape[1]

        # fit ml model
        if isinstance(self.mlModel, Iterable) and self.modelType != 'randomForest':
            for m in self.mlModel:
                m.fit(processed, y_true)
        else:
            self.mlModel.fit(processed, y_true)

    def predict(self, samples, aggregateEnvironment=False, returnScores=False, **kwargs):
        if not isinstance(samples, Iterable):
            samples = [samples]

        # get feature data from prediction samples
        scores, featureData = [], []
        for s in samples:
            try:
                returnDict = self.align(s, returnScore=True, **kwargs)  # TODO: align to cleaned HP?
                alignedPharmacophore, score = returnDict['alignedPharmacophore'], returnDict['score']
                features = self.getFeatureData(alignedPharmacophore, aggregateEnvironment=aggregateEnvironment, **kwargs)
            except AlignmentError:
                score = 0
                features = [0]*self.cleanedHP.numFeatures

            featureData.append(features)
            scores.append(score)

        # predict activity for samples
        featureData = np.stack(featureData, axis=0)
        processed = self.postProcessFeatureData(featureData, training=False, **kwargs)

        # if processed.shape[1] == self.trainingDim:
        if processed.shape[1] > 0:
            if isinstance(self.mlModel, Iterable) and self.modelType != 'randomForest':
                y_pred = []
                for m in self.mlModel:
                    predictions = m.predict(processed)
                    y_pred.append(predictions)
                y_pred = np.sum(np.stack(y_pred, axis=0), axis=0).flatten()

            else:
                y_pred = self.mlModel.predict(processed).flatten()

            # handle unaligned samples -> set predictions to zero
            y_pred = np.where((scores == 0), np.zeros(len(scores)), y_pred)

        else:
            y_pred = np.zeros(len(scores))

        if returnScores:
            return y_pred, scores
        else:
            return y_pred

    def postProcessFeatureData(self, featureData, training=False, **kwargs):
        if training:
            from Data_utils import cross_correlation_filter, low_variance_filter

            varianceFiltered, varianceMask = low_variance_filter(featureData, 0.1)
            correlationFiltered, correlationMask = cross_correlation_filter(varianceFiltered, 0.9)
            self.varianceMask, self.correlationMask = varianceMask, correlationMask

        else:
            varianceFiltered = featureData[:, self.varianceMask]
            correlationFiltered = varianceFiltered[:, self.correlationMask]
            # if correlationFiltered.shape[1] != self.trainingDim:
            #     print(featureData.shape, correlationFiltered.shape, self.trainingDim)

        return correlationFiltered

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

                else:
                    raise TypeError('weightType not recognized')

                features.append(weight)
        return features

    def _align(self, samples, **kwargs):
        samples = self.onAlignmentStart(samples, **kwargs)
        for s in samples:
            self.samples.append(s)
            try:
                returnDict = self.align(s, returnScore=True, **kwargs)
                alignedPharmacophore, score = returnDict['alignedPharmacophore'], returnDict['score']
                self.alignedSamples.append((alignedPharmacophore, score))
                self.nrSamples += 1
                self.merge(alignedPharmacophore)
            except AlignmentError:
                pass
        self.onAlignmentEnd(samples, **kwargs)

    def _trainHPModel(self, samples, **kwargs):
        samples = self.onMLTrainingStart(samples, **kwargs)
        self.trainHPModel(**kwargs)
        self.onMLTrainingEnd(samples, **kwargs)

    def align(self, sample, returnScore=False, returnTfMatrix=False, **kwargs):
        """
        Align the given sample to the HP's model template. Keep in mind that the sample will not be aligned to the
        HP itself, but only its starting point. This has several  advantages:
        - speeds up training, since we only have to align molecules once and not again after changing some features
        - makes the alignment and HP creation process deterministic, since it no longer depends on the order of the
        samples as in the previous version (see pharmacophores_3.py).
        :param sample:
        :param kwargs:
        :return:
        """
        if isinstance(sample, Chem.BasicMolecule):
            returnDict = self._alignMolecule(sample, returnScore=returnScore, returnTfMatrix=returnTfMatrix, **kwargs)
        elif isinstance(sample, Pharm.BasicPharmacophore):
            returnDict = self._alignPharmacophore(sample, returnScore=returnScore, returnTfMatrix=returnTfMatrix, **kwargs)
        else:
            raise TypeError('Given type {} not known. Type should be one of [Chem.BasicMolecule, Pharm.BasicPharmacophore].'.format(type(sample)))

        alignedPharmacophore = returnDict['alignedPharmacophore']
        a = sample.getProperty(LOOKUPKEYS['activity'])
        alignedPharmacophore.setProperty(LOOKUPKEYS['activity'], a)

        return returnDict

    def _alignMolecule(self, mol, returnScore=True, returnAllConformations=False, returnTfMatrix=False,
                       returnConfNr=False, **kwargs):
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
        bestTfMatrix = Math.Matrix4D()
        bestConf = 0
        ref = kwargs.get('reference', self)
        self.aligner.addFeatures(ref, True)

        for j in range(Chem.getNumConformations(mol)):
            Chem.applyConformation(mol, j)
            p = get_pharmacophore(mol, fuzzy=self.fuzzy)
            self.aligner.addFeatures(p, False)

            signal.alarm(10)
            try:
                while self.aligner.nextAlignment():
                    tfMatrix = self.aligner.getTransform()
                    score = self.scorer(ref, p, tfMatrix)

                    if score is not None:
                        Pharm.transform3DCoordinates(p, tfMatrix)
                        conformations.append((p, score))
                        if score > bestScore:
                            bestScore = score
                            bestPharmacophore = p
                            bestTfMatrix.assign(tfMatrix)
                            bestConf = j
                    else:
                        conformations.append((p, score))
                signal.alarm(0)
            except TimeoutError:
                raise AlignmentError

            self.aligner.clearEntities(False)
        self.aligner.clearEntities(True)

        if bestScore == 0:
            raise AlignmentError

        return {
            'alignedPharmacophore': bestPharmacophore,
            'score': bestScore if returnScore else None,
            'conformations': conformations if returnAllConformations else None,
            'tfMatrix': bestTfMatrix if returnTfMatrix else None,
            'confNr': bestConf if returnConfNr else None
        }

    def _alignPharmacophore(self, p, returnScore=True,  returnTfMatrix=False, **kwargs):
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

        ref = kwargs.get('reference', self)
        self.aligner.addFeatures(ref, True)
        self.aligner.addFeatures(p, False)
        bestScore = 0
        bestTfMatrix = Math.Matrix4D()
        signal.alarm(5)  # in seconds
        try:
            while self.aligner.nextAlignment():
                tfMatrix = self.aligner.getTransform()
                score = self.scorer(ref, p, tfMatrix)

                if score is not None:
                    if score > bestScore:
                        Pharm.transform3DCoordinates(p, tfMatrix)
                        bestScore = score
                        bestTfMatrix.assign(tfMatrix)
            signal.alarm(0)
        except TimeoutError:
            raise AlignmentError

        self.aligner.clearEntities(False)
        self.aligner.clearEntities(True)

        if bestScore == 0:
            raise AlignmentError

        return {
            'alignedPharmacophore': p,
            'score': bestScore if returnScore else None,
            'conformations': None,
            'tfMatrix': bestTfMatrix if returnTfMatrix else None,
            'confNr': 0
        }

    def fuzzyfyPharmacophore(self, p, **kwargs):
        for f in p:
            if Pharm.getType(f) == 5 or Pharm.getType(f) == 6:
                Pharm.clearOrientation(f)
                Pharm.setGeometry(f, Pharm.FeatureGeometry.SPHERE)

    def cleanFeatures(self, **kwargs):
        """
        Remove ambiguous features.
        Each feature should at least have two entries --> one merged feature.
        The range of activity values for each feature should not be more than half the entire range of feature activity
        of all features.
        :param kwargs:
        :return:
        """
        # determine min and max value of activity.
        toRemove = []
        startingActivities = self.getFeature(0).getProperty(LOOKUPKEYS['activities'])
        refMin, refMax = min(startingActivities), max(startingActivities)
        storedActivities = []
        for i in range(self.numFeatures):
            f = self.getFeature(i)
            activities = np.array(f.getProperty(LOOKUPKEYS['activities']))
            added = self.cleanedHP.addFeature()  # copy features to cleaned HP, so we keep the original in self
            added.assign(f)
            if len(activities) == 1:
                toRemove.append(i)  # remove features with only one feature activity
                continue
            else:
                storedActivities.append(activities)  # store activities, so we do not have to iterate the features again

            if min(activities) < refMin:
                refMin = min(activities)
            if max(activities) > refMax:
                refMax = max(activities)

        for i in reversed(toRemove):
            self.cleanedHP.removeFeature(i)

        # query the features for their range
        toRemove = []
        for i, activities in enumerate(storedActivities):
            if len(activities) > 2:
                lessThan75 = (activities <= np.quantile(activities, 0.75))
                biggerThan25 = (activities >= np.quantile(activities, 0.25))
                narrowedSubset = activities[lessThan75 * biggerThan25]
                minValue, maxValue = np.min(narrowedSubset), np.max(narrowedSubset)

            else:
                minValue, maxValue = min(activities), max(activities)

            if maxValue-minValue > (refMax-refMin)/2:
                toRemove.append(i)
        for i in reversed(toRemove):
            self.cleanedHP.removeFeature(i)

    def onMLTrainingStart(self, samples, **kwargs):
        self.cleanFeatures()

        if self.weightType == 'gaussian':  # create gaussian features so we can calculate distance easy
            from ShapeAlignment import prepareForShapeAlignment

            for f in self.cleanedHP:
                featureSet = Pharm.FeatureSet()
                featureSet.addFeature(f)
                featureShape, featureShapeFunction = prepareForShapeAlignment(featureSet)
                self.gaussianFeatures.append((featureShape, featureShapeFunction))

        return samples

    def onMLTrainingEnd(self, samples, **kwargs):
        pass

    def onAlignmentStart(self, samples, **kwargs):
        return samples

    def onAlignmentEnd(self, samples, **kwargs):
        pass

    def onFitStart(self, samples, **kwargs):
        shuffle(samples)
        return samples

    def onFitEnd(self, samples, **kwargs):
        pass

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

        save_pharmacophore(self.cleanedHP, '{}cleanedModel.pml'.format(path))
        save_pharmacophore(self, '{}entireModel.pml'.format(path))
        if isinstance(self.mlModel, Iterable) and self.modelType != 'randomForest':
            os.mkdir('{}mlModel/'.format(path))
            for i, m in enumerate(self.mlModel):
                with open('{}mlModel/model{}.pkl'.format(path, str(i)), 'wb') as mlfile:
                    pickle.dump(m, mlfile)
        else:
            with open('{}mlModel.pkl'.format(path), 'wb') as mlfile:
                pickle.dump(self.mlModel, mlfile)
        parameters = {
            'fuzzy': self.fuzzy,
            'weightType': self.weightType,
            'logPath': self.logPath,
            'name': self.name,
            'alignmentTimeout': self.alignmentTimeout,
            'maxDistance': self.maxDistance,
            'nrSamples': self.nrSamples,
            'trainingDim': self.trainingDim,
            'varianceMask': self.varianceMask.tolist(),
            'correlationMask': self.correlationMask.tolist(),
        }
        with open('{}parameters.json'.format(path), 'w') as f:
            json.dump(parameters, f, indent=2)

    def load(self, path, **kwargs):
        import pickle
        from Pharmacophore_tools import load_pml_pharmacophore

        with open('{}parameters.json'.format(path), 'r') as f:
            parameters = json.load(f)
        for key, value in parameters.items():
            setattr(self, key, value)

        print('Loading HP model from {}'.format(path))

        # load hp model
        if os.path.isfile('{}cleanedModel.pml'.format(path)):
            self.cleanedHP = load_pml_pharmacophore('{}cleanedModel.pml'.format(path))
        else:
            print('Could not load hp model at file: cleanedModel.pml')
        if os.path.isfile('{}entireModel.pml'.format(path)):
            p = load_pml_pharmacophore('{}entireModel.pml'.format(path))
            for f in p:
                added = self.addFeature()
                added.assign(f)
        else:
            print('Could not find uncleaned model at file: entireModel.pml')

        # load ml model
        if os.path.isdir('{}mlModel'.format(path)):
            mlModel = []
            for f in os.path.isdir('{}mlModel'.format(path)):
                with open('{}mlModel/{}'.format(path, f), 'rb') as mlfile:
                    mlModel.append(pickle.load(mlfile))
            self.mlModel = mlModel
        elif os.path.isfile('{}mlModel.pkl'.format(path)):
            with open('{}mlModel.pkl'.format(path), 'rb') as mlfile:
                self.mlModel = pickle.load(mlfile)
        else:
            print('Could not find ml model at file: mlModel.pkl or folder: mlModel/')


class OrthogonalHyperpharmacophore(SequentialHyperpharmacophore):

    """
    The idea behind this HP model is the sequential update / improvement of your model by taking into consideration,
    where the model has the least information or is the least unsure about.
    The basic algorithm works the following:
    - initiation of HP by template
    - chosing least compatible sample from set of training samples as determined by alignment score (if more than
    one sample cannot be aligned, the first sample without alignment will be chosen and set as a second 'template'.
    Following samples are then aligned to both templates. If they are alignable to just one template, it is added to
    the respective collection. However, if both templates can be aligned too it, then a common denominator between the
    previously not compatible pharmacophore sets is found on which both sets are aligned and merged.
    - Samples aligned are merged to the HP as is done by the SequentialHyperpharmacophore.
    - Once all samples were aligned, features are cleaned and prepared for prediction by post-processing as adopted
    from the SequentialHP.
    """

    def __init__(self,
                 template=None,
                 modelType='linearRegression',
                 modelKwargs=None,
                 logPath=None,
                 weightType='distance',
                 name='OrthogonalHyperpharmacophore',
                 alignmentTimeout=30,
                 fuzzy=True,
                 **kwargs
                 ):
        super(OrthogonalHyperpharmacophore, self).__init__(None,  # set template to None. Init separetely here
                                                           modelType=modelType,
                                                           modelKwargs=modelKwargs,
                                                           logPath=logPath,
                                                           weightType =weightType,
                                                           name=name,
                                                           alignmentTimeout=alignmentTimeout,
                                                           fuzzy=fuzzy,
                                                           **kwargs
                                                           )

        self.template = []
        self.initKwargs = {
            'alignmentTimeout': alignmentTimeout,
            'fuzzy': fuzzy,
        }  # remaining parameters are not important
        if template is not None:   # init template
            if isinstance(template, Pharm.BasicPharmacophore):
                self.template.append(SequentialHyperpharmacophore(template, **self.initKwargs))
            elif isinstance(template, (list, tuple)):
                self.template.append(SequentialHyperpharmacophore(template[:2], **self.initKwargs))

    def initTemplate(self, samples, **kwargs):
        if isinstance(samples[0], Pharm.BasicPharmacophore):
            self.template.append(SequentialHyperpharmacophore(samples[0], **self.initKwargs))
            samples = samples[1:]
        else:  # assume molecules
            self.template.append(SequentialHyperpharmacophore(samples[:2], **self.initKwargs))
            samples = samples[2:]

        return samples

    def findWorstAlignment(self, samples, **kwargs):
        """
        Finds the least favorable sample from a given list of samples. However, is more than one template, it finds
        the template which merges the most templates. If there are more than one templates, but sample can only be
        aligned to one sample, once again it finds the sample with the least favorable fit.
        :param samples:
        :param kwargs:
        :return:
        """
        # align samples to template
        if len(self.template) == 1:  # only one HP exists so far -> standard alignment from SequentialHP
            worstAlignment = 1  # alignmentScore
            chosenSample = None
            sampleIndex = 0
            isAlignable = False
            alignableToTemplates = []

            for sIndex, sample in enumerate(samples):

                try:
                    returnDict = self.template[0].align(sample, returnScore=True, **kwargs)
                    alignedSample, score = returnDict['alignedPharmacophore'], returnDict['score']

                    if score < worstAlignment:
                        worstAlignment = score
                        chosenSample = sample
                        sampleIndex = sIndex
                        isAlignable = True
                        alignableToTemplates = [0]
                        # self.template[0].merge(alignedSample)
                    # self.template[0]._align([sample], **kwargs)  # already takes care of merging etc
                except AlignmentError:  # no alignment -> immediately return
                    return False, [sample, []], [samples[i] for i in range(len(samples)) if i != sampleIndex]

            return isAlignable, [chosenSample, alignableToTemplates], [samples[i] for i in range(len(samples)) if i != sampleIndex]

        else:  # we have multiple templates which are not yet compatible
            alignableSamples = []
            for sIndex, sample in enumerate(samples):

                alignableConformations = {}
                for sampleConf in range(Chem.getNumConformations(sample)):
                    Chem.applyConformation(sample, sampleConf)
                    samplePharmacophore = get_pharmacophore(sample, fuzzy=self.fuzzy)

                    alignableToTemplates = []
                    for i, t in enumerate(self.template):
                        if isinstance(t, Pharm.BasicPharmacophore):  # simple alignment -> sequential HP
                            try:
                                score = t._alignPharmacophore(samplePharmacophore, **kwargs)['score']
                                alignableToTemplates.append((i, score))  # store template index as well as score
                            except AlignmentError:
                                pass
                        else:  # template is still a molecule
                            bestScore = 0
                            for confNr in range(Chem.getNumConformations(t)):
                                Chem.applyConformation(t, confNr)
                                p = get_pharmacophore(t, fuzzy=self.fuzzy)
                                try:
                                    score = self._alignPharmacophore(samplePharmacophore, reference=p, **kwargs)['score']
                                    if score > bestScore:
                                        bestScore = score
                                except AlignmentError:
                                    pass

                            if bestScore > 0:
                                alignableToTemplates.append((i, bestScore))
                            else:
                                pass
                    if len(alignableToTemplates) > 0:
                        alignableConformations[sampleConf] = alignableToTemplates

                alignableSamples.append(alignableConformations)

            # check if we found an alignment for any sample
            foundAlignment = False
            for alignedSample in alignableSamples:
                if len(alignedSample) > 1:
                    foundAlignment = True
                    break
            if not foundAlignment:  # no alignment could be found to any template for any sample--> return immediately
                return False, [samples[0], []], samples[1:]  # just return first sample, since none could be aligned

            # samples were aligned. Now find the conformation which aligns to the most templates.
            alignableToNTemplates = 1  # at least alignable to one template per default
            mostAlignedSample = None  # store tuple of sample, sampleConformation, and sampleIndex
            alignableTemplates = None
            for sampleNr, alignedConformations in enumerate(alignableSamples):
                if len(alignedConformations) == 0:
                    continue

                for conf, alignedTemplates in alignedConformations.items():
                    if len(alignedTemplates) > alignableToNTemplates:
                        alignableToNTemplates = len(alignedTemplates)
                        mostAlignedSample = (samples[sampleNr], conf, sampleNr)
                        alignableTemplates = alignedTemplates

            if alignableToNTemplates > 1:
                sample, conf, sampleNr = mostAlignedSample
                Chem.applyConformation(sample, conf)

                return True, [sample, alignableTemplates], [samples[i] for i in range(len(samples)) if i != sampleNr]
            else:  # could only be aligned to a single template --> find sample with worst alignment

                worstAlignment = 1
                chosenSample = None
                sampleIndex = 0
                # isAlignable = False
                alignableToTemplates = []

                for sIndex, sample in enumerate(samples):

                    for tIndex, template in self.template:
                        if isinstance(template, SequentialHyperpharmacophore):
                            try:
                                returnDict = template.align(sample, **kwargs, returnScore=True)
                                alignedSample, score = returnDict['alignedPharmacophore'], returnDict['score']

                                if score < worstAlignment:
                                    worstAlignment = score
                                    chosenSample = sample
                                    sampleIndex = sIndex
                                    # isAlignable = True
                                    alignableToTemplates[0] = tIndex
                            except AlignmentError:
                                pass
                        else:
                            for conf in range(Chem.getNumConformations(template)):
                                Chem.applyConformation(template, conf)
                                p = get_pharmacophore(template, fuzzy=self.fuzzy)

                                try:
                                    returnDict = self._alignMolecule(sample, reference=p, returnScore=True, **kwargs)
                                    alignedSample, score = returnDict['alignedPharmacophore'], returnDict['score']

                                    if score < worstAlignment:
                                        worstAlignment = score
                                        chosenSample = sample
                                        sampleIndex = sIndex
                                        # isAlignable = True
                                        alignableToTemplates[0] = tIndex

                                except AlignmentError:
                                    pass

                return True, [chosenSample, alignableToTemplates], [samples[i] for i in range(len(samples)) if i != sampleIndex]

    def mergeTemplates(self, sample, templateIndices, **kwargs):
        # sample is already aligned!
        p = get_pharmacophore(sample, fuzzy=self.fuzzy)
        newTemplate = SequentialHyperpharmacophore(p, **self.initKwargs)
        for i in templateIndices:
            print(templateIndices, i, type(i))
            template = self.template[i]

            try:
                returnDict = newTemplate.align(template,  returnScore=True, returnTfMatrix=True, **kwargs)
                alignedSample, score, tf = returnDict['alignedPharmacophore'], returnDict['score'], returnDict['tfMatrix']
                if isinstance(template, SequentialHyperpharmacophore):
                    oldNrSamples = template.nrSamples
                    templateSamples = []
                    for s in template.samples:
                        Pharm.transform3DCoordinates(s, tf)
                        templateSamples.append(s)
                    alignedSamples = []
                    for s in template.alignedSamples:
                        Pharm.transform3DCoordinates(s, tf)
                        alignedSamples.append(s)
                else:
                    oldNrSamples = 0
                    templateSamples, alignedSamples = [], []

                newTemplate.merge(alignedSample)
                newTemplate.nrSamples = newTemplate.nrSamples + 1 + oldNrSamples
                newTemplate.samples.append(sample)
                newTemplate.alignedSamples.append(alignedSample)
                newTemplate.samples.extend(templateSamples)
                newTemplate.alignedSamples.extend(alignedSamples)
            except AlignmentError:  # cannot happen, since we aligned it before already
                pass

        # delete template no longer needed
        sortedTemplateIndices = reversed(sorted(templateIndices))
        for i in sortedTemplateIndices:
            del self.template[i]

        # add new template to list
        self.template.append(newTemplate)

    def fit(self, samples, **kwargs):
        if not isinstance(samples, Iterable):
            samples = [samples]

        samples = self.onFitStart(samples, **kwargs)

        if len(self.template) == 0:  # init template
            samples = self.initTemplate(samples, **kwargs)  # consumes one or more samples and returns the remaining

        while True:
            isAlignable, sample, samples = self.findWorstAlignment(samples, **kwargs)
            sample, templateIndices = sample
            if not isAlignable:  # no alignment was found
                if isinstance(sample, Chem.BasicMolecule):
                    self.template.append(sample)
                else:  # assume pharmacophore
                    self.template.append(SequentialHyperpharmacophore(sample, **self.initKwargs))
            else:  # found alignment, but align once again to chosen template
                if len(templateIndices) > 1:
                    self.mergeTemplates(sample, templateIndices)
                else:
                    templateIndices = templateIndices[0]
                    t = self.template[templateIndices]
                    if isinstance(t, SequentialHyperpharmacophore):
                        t._align([sample], **kwargs)
                    else:
                        self.template[templateIndices] = SequentialHyperpharmacophore([self.template[templateIndices], sample], **self.initKwargs)

            if len(samples) == 0:
                break

        # select best template
        if len(self.template) > 1:  # no common ground between the templates was found -> select one
            mostSamplesAligned = 0
            bestTemplate = self.template[0]  # first is default
            for t in self.template:
                if not isinstance(t, SequentialHyperpharmacophore):
                    continue
                if t.nrSamples > mostSamplesAligned:
                    mostSamplesAligned = t.nrSamples
                    bestTemplate = t
        else:
            bestTemplate = self.template[0]

        # finally init self from template
        self.nrSamples = bestTemplate.nrSamples
        self.samples = bestTemplate.samples
        self.alignedSamples = bestTemplate.alignedSamples
        for f in bestTemplate:
            addedFeature = self.addFeature()
            addedFeature.assign(f)

        # train ML model
        self._trainHPModel(samples, **kwargs)

        self.onFitEnd(samples, **kwargs)


class OrthogonalHyperpharmacophore2(OrthogonalHyperpharmacophore):

    """
    Same principle as in the first version is used, with the distinction that to aligned samples are choosen based on
    predicted activity. The samples to be aligned are the ones with the highest error and / or the least confidence of
    the model in the prediction, since apparently these samples contain information currently not reflected by the
    model.
    Due to the high amount of training models and predicting samples during the fitting phase, an ML model or
    mechanism for prediction should be choosen that is simple and has low complexity. One the one hand this will greatly
    improve speed, but more importantly, at the beginning only a few samples are available to learn from, leading to
    most likely overfitting of the models to the samples. Therefore, very basic models should be chosen.
    """

    def __init__(self,
                 template,
                 modelType='linearRegression',
                 modelKwargs=None,
                 logPath=None,
                 weightType='distance',
                 name='OrthogonalHyperpharmacophore',
                 alignmentTimeout=30,
                 fuzzy=True,
                 **kwargs
                 ):
        super(OrthogonalHyperpharmacophore2, self).__init__(template,
                                                            modelType=modelType,
                                                            modelKwargs=modelKwargs,
                                                            logPath=logPath,
                                                            weightType=weightType,
                                                            name=name,
                                                            alignmentTimeout=alignmentTimeout,
                                                            fuzzy=fuzzy,
                                                            **kwargs
                                                            )
