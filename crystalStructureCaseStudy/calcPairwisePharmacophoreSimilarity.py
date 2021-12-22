"""
For two given datasets (or pharmacophores) calculate the pairwise similarity between these two. You can choose between
two options:
    - alignment score: similarity is defined by the alignment score between the two entities
    - feature score: similarity is determined by the feature score as described in https://pubs.acs.org/doi/abs/10.1021/ci200060s
"""
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Union
import logging

import numpy as np
import pandas as pd
import CDPL.Chem as Chem
import CDPL.Pharm as Pharm
import CDPL.Math as Math

from src.molecule_tools import SDFReader
from src.pharmacophore_tools import loadPharmacophore, getPharmacophore


FEATURE_TYPES = [
    Pharm.FeatureType.AROMATIC,
    Pharm.FeatureType.HYDROPHOBIC,
    Pharm.FeatureType.H_BOND_DONOR,
    Pharm.FeatureType.H_BOND_ACCEPTOR,
    Pharm.FeatureType.NEG_IONIZABLE,
    Pharm.FeatureType.POS_IONIZABLE,
]
DEFAULT_TOLERANCE = 1.5


def parseArgs():
    parser = ArgumentParser()
    parser.add_argument('-inputA', required=True, type=str, help='path to pml or sdf file')
    parser.add_argument('-inputB', required=False, type=str, default=None,
                        help='path to pml or sdf file. if not given, inputA will be used')
    parser.add_argument('-similarityScore', required=False, type=str, default='alignment',
                        help='one of ["alignment", "feature"], if not provided, then "alignment" is default')
    parser.add_argument('-fileName', required=True, type=str, default='name of output file')
    parser.add_argument('-tolerance', required=False, type=float, default=0.8, help='overlap tolerance')
    return parser.parse_args()


class Scorer:

    def __init__(self,
                 similarityMetric: str,
                 tolerance: float = None,
                 ):
        self.similarityMetric = similarityMetric
        self.tolerance = tolerance if tolerance is not None else DEFAULT_TOLERANCE

    def calculateSimilarity(self,
                            elementA: Union[Chem.BasicMolecule, Pharm.BasicPharmacophore],
                            elementB: Union[Chem.BasicMolecule, Pharm.BasicPharmacophore],
                            ) -> float:
        alignedPharmacophores, alignmentScore = alignElements(elementA, elementB)
        if self.similarityMetric == 'alignment':
            return alignmentScore

        elif self.similarityMetric == 'feature':
            return self.calculateFeatureScore(*alignedPharmacophores)

        else:
            raise ValueError(
                'Invalid similarity metric "{}" provided. Most be one of [alignment, feature]'.format(self.similarityMetric))

    def calcEuclideanDistance(self, coordsA: np.array, coordsB: np.array) -> float:
        return np.sqrt(np.sum(np.power(coordsA - coordsB, 2), axis=0))

    def calculateExponent(self, distance) -> float:
        return -2.5 * (distance / self.tolerance) ** 2

    def calculateFeatureScore(self,
                              pharmacophoreA: Pharm.BasicPharmacophore,
                              pharmacophoreB: Pharm.BasicPharmacophore,
                              ) -> float:
        fab, fa, fb = 0, 0, 0
        for featureType in FEATURE_TYPES:

            for i in range(pharmacophoreA.numFeatures):
                featureA = pharmacophoreA.getFeature(i)

                if Pharm.getType(featureA) != featureType:
                    continue

                fa += np.exp(self.calculateExponent(0))

                for j in range(pharmacophoreB.numFeatures):
                    featureB = pharmacophoreB.getFeature(j)

                    if Pharm.getType(featureB) != featureType:
                        continue

                    dij = self.calcEuclideanDistance(Chem.get3DCoordinates(featureA).toArray(),
                                                     Chem.get3DCoordinates(featureB).toArray())
                    fab += np.exp(self.calculateExponent(dij))

        for j in range(pharmacophoreB.numFeatures):
            fb += np.exp(self.calculateExponent(0))

        return fab / np.sqrt(fa * fb)


def loadDataset(path: str) -> List[Union[Chem.BasicMolecule, Pharm.BasicPharmacophore]]:
    if path.endswith('.pml'):
        logging.info('Loading pharmacophore from {}'.format(path))
        return [loadPharmacophore(path)]
    elif path.endswith('.sdf'):
        logging.info('Loading molecules from {}'.format(path))
        r = SDFReader(path, multiconf=True)
        return [mol for mol in r]
    else:
        raise ValueError('Invalid input file given: {} given but must be one of [.pml, .sdf]'.format(path[-4:]))


def loadDatasets(pathA: str,
                 pathB: str,
                 ) -> Tuple[List[Union[Chem.BasicMolecule, Pharm.BasicPharmacophore]],
                            List[Union[Chem.BasicMolecule, Pharm.BasicPharmacophore]]]:
    datasetA = loadDataset(pathA)
    datasetB = loadDataset(pathB)
    return datasetA, datasetB


def alignMoleculeToMolecule(molA: Chem.BasicMolecule,
                            molB: Chem.BasicMolecule,
                            ) -> Tuple[Tuple[Pharm.BasicPharmacophore, Pharm.BasicPharmacophore], float]:
    bestScore = 0
    bestTfMatrix = Math.Matrix4D()
    aligner = Pharm.PharmacophoreAlignment(True)
    scorer = Pharm.PharmacophoreFitScore()
    bestConfA, bestConfB = 0, 0

    for i in range(Chem.getNumConformations(molA)):
        Chem.applyConformation(molA, i)
        pharmA = getPharmacophore(molA, fuzzy=True)

        aligner.addFeatures(pharmA, True)
        for j in range(Chem.getNumConformations(molB)):
            Chem.applyConformation(molB, j)
            pharmB = getPharmacophore(molB, fuzzy=True)
            aligner.addFeatures(pharmB, False)

            # assignment of tf matrix and score comparison happens inside alignment
            alignedPharmacophores, score = alignPharmacophoreToPharmacophore(pharmA,
                                                                             pharmB,
                                                                             aligner=aligner,
                                                                             scorer=scorer,
                                                                             bestTfMatrix=bestTfMatrix,
                                                                             )
            if score > bestScore:
                bestScore = score
                bestConfA = i
                bestConfB = j

            aligner.clearEntities(False)

        aligner.clearEntities(True)

    Chem.applyConformation(molA, bestConfA)
    Chem.applyConformation(molB, bestConfB)
    pharmA, pharmB = getPharmacophore(molA, fuzzy=True), getPharmacophore(molB, fuzzy=True)
    Pharm.transform3DCoordinates(pharmB, bestTfMatrix)
    return (pharmA, pharmB), bestScore


def alignMoleculeToPharmacophore(pharm: Pharm.BasicPharmacophore,
                                 mol: Chem.BasicMolecule,
                                 ) -> Tuple[Tuple[Pharm.BasicPharmacophore, Pharm.BasicPharmacophore], float]:
    bestScore = 0
    bestTfMatrix = Math.Matrix4D()
    aligner = Pharm.PharmacophoreAlignment(True)
    scorer = Pharm.PharmacophoreFitScore()
    bestConf = 0

    aligner.addFeatures(pharm, True)
    for j in range(Chem.getNumConformations(mol)):
        Chem.applyConformation(mol, j)
        pharmB = getPharmacophore(mol, fuzzy=True)
        aligner.addFeatures(pharmB, False)

        # assignment of tf matrix and score comparison happens inside alignment
        alignedPharmacophores, score = alignPharmacophoreToPharmacophore(pharm,
                                                                         pharmB,
                                                                         aligner=aligner,
                                                                         scorer=scorer,
                                                                         bestTfMatrix=bestTfMatrix,
                                                                         )
        if score > bestScore:
            bestScore = score
            bestConf = j

        aligner.clearEntities(False)

    Chem.applyConformation(mol, bestConf)
    pharmB = getPharmacophore(mol, fuzzy=True)
    Pharm.transform3DCoordinates(pharmB, bestTfMatrix)
    return (pharm, pharmB), bestScore


def alignPharmacophoreToPharmacophore(pharmA: Pharm.BasicPharmacophore,
                                      pharmB: Pharm.BasicPharmacophore,
                                      aligner: Pharm.PharmacophoreAlignment = None,
                                      scorer: Pharm.PharmacophoreFitScore = None,
                                      bestTfMatrix: Math.Matrix4D = None,
                                      bestScore: float = None,
                                      transformCoordinates: bool = False
                                      ) -> Tuple[Tuple[Pharm.BasicPharmacophore, Pharm.BasicPharmacophore], float]:
    if aligner is None:
        aligner = Pharm.PharmacophoreAlignment(True)
        aligner.addFeatures(pharmA, True)
        aligner.addFeatures(pharmB, False)

    scorer = Pharm.PharmacophoreFitScore() if scorer is None else scorer
    bestScore = 0 if bestScore is None else bestTfMatrix
    bestTfMatrix = Math.Matrix4D() if bestTfMatrix is None else bestTfMatrix
    while aligner.nextAlignment():
        tfMatrix = aligner.getTransform()
        score = scorer(pharmA, pharmB, tfMatrix)

        if score is not None:
            if score > bestScore:
                Pharm.transform3DCoordinates(pharmB, tfMatrix)
                bestScore = score
                bestTfMatrix.assign(tfMatrix)

    if transformCoordinates:
        Pharm.transform3DCoordinates(pharmB, bestTfMatrix)
    return (pharmA, pharmB), bestScore


def alignElements(elementA: Union[Chem.BasicMolecule, Pharm.BasicPharmacophore],
                  elementB: Union[Chem.BasicMolecule, Pharm.BasicPharmacophore],
                  ) -> Tuple[Tuple[Pharm.BasicPharmacophore, Pharm.BasicPharmacophore], float]:
    if isinstance(elementA, Chem.BasicMolecule) and isinstance(elementB, Chem.BasicMolecule):
        pharmacophores, alignmentScore = alignMoleculeToMolecule(elementA, elementB)

    elif isinstance(elementA, Pharm.BasicPharmacophore) and isinstance(elementB, Pharm.BasicPharmacophore):
        pharmacophores, alignmentScore = alignPharmacophoreToPharmacophore(elementA,
                                                                           elementB,
                                                                           transformCoordinates=True)

    else:
        if isinstance(elementA, Chem.BasicMolecule):
            pharmacophores, alignmentScore = alignMoleculeToPharmacophore(elementB, elementA)
        else:
            pharmacophores, alignmentScore = alignMoleculeToPharmacophore(elementA, elementB)

    return pharmacophores, alignmentScore


def processDatasets(datasetA: List[Pharm.BasicPharmacophore],
                    datasetB: List[Pharm.BasicPharmacophore],
                    similarityMetric: str,
                    tolerance: float = None
                    ) -> np.array:
    scorer = Scorer(similarityMetric=similarityMetric, tolerance=tolerance)
    similarities = np.ones((len(datasetA), len(datasetB)))
    logging.info('Calculating {} pairwise similarities for {} and {} elements'.format(similarities.size,
                                                                                      similarities.shape[0],
                                                                                      similarities.shape[1],
                                                                                      ))
    for i in range(len(datasetA)):
        elementA = datasetA[i]

        for j in range(len(datasetB)):
            if i == j:
                similarities[i, j] = 1
                continue

            elementB = datasetB[j]

            logging.info('Calculate similarity for {}, {}'.format(i, j))
            similarityScore = scorer.calculateSimilarity(elementA, elementB)
            similarities[i, j] = similarityScore

    return similarities


if __name__ == '__main__':
    args = parseArgs()
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    datasetA, datasetB = loadDatasets(args.inputA, args.inputA if args.inputB is None else args.inputB)

    similarities = processDatasets(datasetA[:5], datasetB[:5], args.similarityScore)
    pd.DataFrame(similarities).to_csv('./{}.csv'.format(args.fileName))
