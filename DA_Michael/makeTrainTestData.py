import logging
from argparse import ArgumentParser
from typing import List, Tuple
import os

import CDPL.Chem as Chem
import numpy as np

from src.molecule_tools import SDFReader, mol_to_sdf
from src.qphar import LOOKUPKEYS


def parseArgs():
    parser = ArgumentParser()
    parser.add_argument('-i', required=True, type=str, help='training set of type sdf')
    parser.add_argument('-activityName', required=True, type=str, help='name of activity property in sdf file')
    parser.add_argument('-validationSize', required=True, type=float, help='fraction of validation data')
    parser.add_argument('-o', required=True, type=str, help='output folder')
    return parser.parse_args()


def loadMolecules(filePath: str, activityName: str) -> List[Chem.BasicMolecule]:
    if not filePath.endswith('.sdf'):
        raise IOError('File cannot be read. {} was provided, but .sdf is required.'.format(os.path.split(filePath)[0]))

    r = SDFReader(filePath, multiconf=True)
    molecules: List[Chem.BasicMolecule] = []
    for mol in r:
        sdb: Chem.StringDataBlock = Chem.getStructureData(mol)
        activity = None
        for entry in sdb:
            if activityName in entry.header:
                try:
                    activity = float(entry.data)
                except ValueError:
                    logging.info('Could not convert property {} to float from molecule {}'.format(activityName,
                                                                                                  Chem.getName(mol)))
                    continue

        if activity is not None:
            mol.setProperty(LOOKUPKEYS['activity'], activity)
            molecules.append(mol)
        else:
            logging.info('No activity value found for molecule {}: skipping'.format(Chem.getName(mol)))

    return molecules


def randomDataSplit(molecules: List[Chem.BasicMolecule],
                    validationSize: float,
                    ) -> Tuple[List[Chem.BasicMolecule], List[Chem.BasicMolecule]]:
    indices = np.arange(len(molecules))
    validationIndices = np.random.choice(indices, size=int(len(indices) * validationSize), replace=False)
    validationMolecules = [molecules[i] for i in validationIndices]
    trainingMolecules = [molecules[i] for i in indices if i not in validationIndices]
    return trainingMolecules, validationMolecules


if __name__ == '__main__':
    args = parseArgs()
    outputFolder = args.o if args.o.endswith('/') else '{}/'.format(args.o)
    molecules = loadMolecules(args.i, args.activityName)
    trainingMolecules, validationMolecules = randomDataSplit(molecules, args.validationSize)
    mol_to_sdf(trainingMolecules, '{}trainingMolecules.sdf'.format(outputFolder), multiconf=True)
    mol_to_sdf(validationMolecules, '{}validationMolecules.sdf'.format(outputFolder), multiconf=True)
