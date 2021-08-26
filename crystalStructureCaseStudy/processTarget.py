import json

import numpy as np
import pandas as  pd
from requests import request
import CDPL.Chem as Chem
import CDPL.Base as Base
import CDPL.Biomol as Biomol
from argparse import ArgumentParser
import os

from crystalStructureCaseStudy.processCrystalStructures import readPDBFromStream, processPDBStructure
from src.molecule_tools import mol_to_sdf


def downloadPDB(pdbCode: str) -> Chem.BasicMolecule:
    r = request('GET', 'https://files.rcsb.org/download/{pdb}.pdb'.format(pdb=pdbCode))
    if r.ok:
        stream = Base.StringIOStream(r.text)
        try:
            pdbStructure = readPDBFromStream(stream)
            return pdbStructure
        except Exception as e:
            print('Failed to read pdb from Stream')
            print(e)
    else:
        print('Failed to download pdbd structure')
        print(r.status_code, r.text)


def processTargetFile(fileName: str, outputFolder: str, fuzzy: bool = True, xvols: bool = True):
    assert fileName.endswith('.xlsx'), 'File must be XLSX-type'
    if not outputFolder.endswith('/'):
        outputFolder = '{}/'.format(outputFolder)

    if not os.path.isdir(outputFolder):
        os.makedirs(outputFolder)

    df = pd.read_excel(fileName)
    activities = {}
    for i, row in df.iterrows():
        print('Processing {}: ...'.format(row['PDB_code']))
        pdbCode, ligandCode, coFactorCodes = row['PDB_code'], row['ligand_code'], row['co_factors']
        activity = -np.log10(row['exp_activity']*10**-9)
        activities[pdbCode] = {'yTrue': activity}

        pdbMol = downloadPDB(pdbCode)
        if pdbMol is None:
            print('Skipping {} due to missing pdb structure'.format(pdbCode))

        output = '{}{}/'.format(outputFolder, pdbCode)
        if not os.path.isdir(output):
            os.makedirs(output)

        processedPDB = processPDBStructure(pdbMol, output, {ligandCode}, fuzzy=fuzzy, exclusionVolumes=xvols)
        cleanedProtein, extractedLigands, _, _ = processedPDB

        # w = Biomol.FilePDBMolecularGraphWriter('{}cleanedProtein.pdb'.format(output, pdbCode))
        # w.write(cleanedProtein)
        # w.close()

        if len(extractedLigands) > 0:
            for ligandCode, ligand in extractedLigands.items():
                sdb = Chem.StringDataBlock()
                sdb.addEntry('<Activity>', str(activity))
                Chem.setStructureData(ligand, sdb)
                Chem.setName(ligand, ligandCode)
            mol_to_sdf([l for l in extractedLigands.values()], '{}ligands.sdf'.format(output))
        else:
            os.removedirs(output)

    # save activities
    with open('{}activities.json'.format(outputFolder), 'w') as f:
        json.dump(activities, f, indent=2)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', type=str, required=True, help='target xlsx file')
    parser.add_argument('-o', type=str, required=True, help='output folder name')
    parser.add_argument('-notFuzzy', default=False, action='store_true', help='deactivate fuzzyness for ph4')
    parser.add_argument('-notXVols', default=False, action='store_true', help='deactivate xvols for ph4')
    args = parser.parse_args()
    processTargetFile(args.i, args.o, fuzzy=not args.notFuzzy, xvols=not args.notXVols)
