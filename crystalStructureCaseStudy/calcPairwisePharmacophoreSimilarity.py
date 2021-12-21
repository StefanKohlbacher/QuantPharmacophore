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


def parseArgs():
    parser = ArgumentParser()
    parser.add_argument('-inputA', required=True, type=str, help='path to pml or sdf file')
    parser.add_argument('-inputB', required=False, type=str, default=None,
                        help='path to pml or sdf file. if not given, inputA will be used')
    parser.add_argument('-similarityScore', required=False, type=str, default='alignment',
                        help='one of ["alignment", "feature"], if not provided, then "alignment" is default')
    parser.add_argument('-fileName', required=True, type=str, default='name of output file')
    return parser.parse_args()


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


def calculateFeatureScore(pharmacophoreA: Pharm.BasicPharmacophore,
                          pharmacophoreB: Pharm.BasicPharmacophore,
                          ) -> float:
    raise NotImplementedError


def alignMoleculeToMolecule(molA: Chem.BasicMolecule,
                            molB: Chem.BasicMolecule,
                            ) -> Tuple[Tuple[Pharm.BasicPharmacophore, Pharm.BasicPharmacophore], float]:
    bestScore = 0
    bestTfMatrix = Math.Matrix4D()
    aligner = Pharm.PharmacophoreAlignment(True)
    scorer = Pharm.PharmacophoreFitScore()
    bestConfA, bestConfB = 0, 0

    for i in range(Chem.getNumConformations(molA)):
        Chem.setConformation(molA, i)
        pharmA = getPharmacophore(molA, fuzzy=True)

        aligner.addFeatures(pharmA, True)
        for j in range(Chem.getNumConformations(molB)):
            Chem.setConformation(molB, j)
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

    Chem.setConformation(molA, bestConfA)
    Chem.setConformation(molB, bestConfB)
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
        Chem.setConformation(mol, j)
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

    Chem.setConformation(mol, bestConf)
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


def calculateSimilarity(elementA: Union[Chem.BasicMolecule, Pharm.BasicPharmacophore],
                        elementB: Union[Chem.BasicMolecule, Pharm.BasicPharmacophore],
                        similarityMetric: str,
                        ) -> float:
    alignedPharmacophores, alignmentScore = alignElements(elementA, elementB)
    if similarityMetric == 'alignment':
        return alignmentScore

    elif similarityMetric == 'feature':
        return calculateFeatureScore(*alignedPharmacophores)

    else:
        raise ValueError(
            'Invalid similarity metric "{}" provided. Most be one of [alignment, feature]'.format(similarityMetric))


def processDatasets(datasetA: List[Pharm.BasicPharmacophore],
                    datasetB: List[Pharm.BasicPharmacophore],
                    similarityMetric: str,
                    ) -> np.array:
    similarities = np.ones((len(datasetA), len(datasetB)))
    logging.info('Calculating {} pairwise similarities for {} and {} elements'.format(similarities.size,
                                                                                      similarities.shape[0],
                                                                                      similarities.shape[1],
                                                                                      ))
    for i in range(len(datasetA)):
        elementA = datasetA[i]

        for j in range(len(datasetB)):
            elementB = datasetB[j]

            logging.info('Calculate similarity for {}, {}'.format(i, j))
            similarityScore = calculateSimilarity(elementA, elementB, similarityMetric)
            similarities[i, j] = similarityScore

    return similarities


if __name__ == '__main__':
    args = parseArgs()
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    datasetA, datasetB = loadDatasets(args.inputA, args.inputA if args.inputB is None else args.inputB)

    similarities = processDatasets(datasetA, datasetB, args.similarityScore)
    pd.DataFrame(similarities).to_csv('./{}.csv'.format(args.fileName))
