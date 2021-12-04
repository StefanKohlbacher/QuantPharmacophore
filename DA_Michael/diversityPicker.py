"""
From a given set of molecules (SMILES and fingerprints), pick a subset of molecules which are dissimilar to each other
at a given threshold. The molecules picked should represent as many molecules as possible to avoid redundancy, but at
the same time be dissimilar to each other. The following strategy will be applied:
    - calculate pair-wise tanimoto similarity
    - determine whether molecules are similar based on given threshold
    - iteratively pick the molecule with the highest connectivity (as defined by number of similar molecules)
    - add this molecule to list of molecules to keep
    - add remaining molecules to list of molecules to remove, since they are already represented by a molecule
    - remove all molecules from set of molecules to evaluate
    - repeat until all molecules are represented at the current threshold
"""
from argparse import ArgumentParser
import os

import pandas as pd
import numpy as np


def parseArgs():
    parser = ArgumentParser()
    parser.add_argument('-i', type=str, required=True, help='input csv file containing smiles and fingerprints')
    parser.add_argument('-o', type=str, required=True, help='output csv file')
    # parser.add_argument('-fpLength', type=int, required=False, default=1024, help='length of fingerprint bitvector')
    # parser.add_argument('-nr', type=int, required=True, help='number of maximum diverse samples to pick')
    parser.add_argument('-similarityThreshold', type=float, required=False, default=0.4,
                        help='threshold to determine whether molecules are similar')
    return parser.parse_args()


def calculateTanimotoSimilarity(fp1: np.array, fp2: np.array) -> float:
    fp1, fp2 = fp1.flatten(), fp2.flatten()
    nom = np.sum(fp1 * fp2)
    denom = np.sum(fp1) + np.sum(fp2) - nom
    return nom / denom


if __name__ == '__main__':
    args = parseArgs()

    if not args.i.endswith('.csv'):
        raise IOError('Found file type {} but expected .csv'.format(os.path.split(args.i)[0]))
    df = pd.read_csv(args.i, engine='python')

    simThreshold = args.similarityThreshold
    if simThreshold < 0 or simThreshold > 1:
        raise ValueError('Got similarity threshold {}, which is out of bounds [0, 1]'.format(simThreshold))

    # calculate pair-wise tanimoto similarity
    tanimotoMatrix = np.zeros((df.shape[0], df.shape[0]))  # we define self-similarity as 0
    fpColumns = [c for c in df.columns.values if 'bitvector' in c]
    for i in range(df.shape[0]):
        fp1 = df.loc[i, fpColumns]
        for j in range(i+1, df.shape[0]):
            fp2 = df.loc[j, fpColumns]
            tanimoto = calculateTanimotoSimilarity(fp1.values, fp2.values)
            tanimotoMatrix[i, j] = tanimoto
            tanimotoMatrix[j, i] = tanimoto

    # pick n most diverse molecules
    initialNrFeatures = tanimotoMatrix.shape[0]
    indexArray = np.arange(initialNrFeatures)
    similarityMatrix = (tanimotoMatrix > simThreshold)  # 2d array of boolean values
    similarMoleculesPerMolecule = np.sum(similarityMatrix, axis=0)
    moleculesToKeep = []  # molecules which are representative of similar molecules
    moleculesToRemove = []  # molecules which are represented by similar molecule
    while similarMoleculesPerMolecule.sum() > 0:  # there still exist molecules which are similar at the given threshold
        highestConnected = np.argmax(similarMoleculesPerMolecule)
        moleculesToKeep.append(highestConnected)
        redundantMolecules = indexArray[similarityMatrix[highestConnected]]
        moleculesToRemove.extend(redundantMolecules.tolist())

        # symbolically remove molecules from similarity matrix
        similarityMatrix[highestConnected, highestConnected] = False
        similarityMatrix[redundantMolecules, :] = False
        similarityMatrix[:, redundantMolecules] = False

        similarMoleculesPerMolecule = np.sum(similarityMatrix, axis=0)

    # save representative molecules
    df.loc[moleculesToKeep, :].drop(fpColumns, axis=1).to_csv(args.o)
