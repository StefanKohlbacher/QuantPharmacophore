import os.path
from argparse import ArgumentParser
import json

import pandas as pd

from src.hyperpharmacophore import DistanceHyperpharmacophore
from src.pharmacophore_tools import loadPharmacophore, savePharmacophore


def loadQpharModel(path: str) -> DistanceHyperpharmacophore:
    model = DistanceHyperpharmacophore()
    model.load(path)
    return model


def main(targetFolder: str, modelPath: str):
    if not targetFolder.endswith('/'):
        targetFolder = '{}/'.format(targetFolder)

    with open('{}activities.json'.format(targetFolder), 'r') as f:
        activities = json.load(f)

    qpharModel = loadQpharModel(modelPath)

    pdbCodes = list(activities.keys())  # make list to have fixed order
    pharmacophores = {
        'interaction': [],
        'ligand': []
    }
    for pdbCode in pdbCodes:
        path = '{}{}/'.format(targetFolder, pdbCode)
        print('Loading', '{}interaction-pharmacophore.pml'.format(path))
        print('Loading', '{}ligand-pharmacophore.pml'.format(path))
        intPharm = loadPharmacophore('{}interaction-pharmacophore.pml'.format(path))
        ligPharm = loadPharmacophore('{}ligand-pharmacophore.pml'.format(path))
        pharmacophores['interaction'].append(intPharm)
        pharmacophores['ligand'].append(ligPharm)

    print('Predicting pharmacophores')
    yPredInt, alignmentScoresInt, mlFeaturesInt, alignedPharmacophoresInt = qpharModel.predict(pharmacophores['interaction'], returnScores=True, returnFeatureData=True, returnAlignedPharmacophores=True)
    yPredLig, alignmentScoresLig, mlFeaturesLig, alignedPharmacophoresLig = qpharModel.predict(pharmacophores['ligand'], returnScores=True, returnFeatureData=True, returnAlignedPharmacophores=True)

    results = pd.DataFrame(index=pdbCodes,
                           columns=['yTrue', 'yPredInt', 'yPredLig', 'alignmentScoreInt', 'alignmentScoreLig'])
    results['yTrue'] = [activities[pdbCode]['yTrue'] for pdbCode in pdbCodes]
    results['yPredInt'] = yPredInt
    results['yPredLig'] = yPredLig
    results['alignmentScoreInt'] = alignmentScoresInt
    results['alignmentScoreLig'] = alignmentScoresLig

    results.to_csv('{}predictions.csv'.format(targetFolder))
    print('Saved results to: ', '{}predictions.csv'.format(targetFolder))

    pd.DataFrame(mlFeaturesInt).to_csv('{}mlFeaturesInt.csv'.format(targetFolder))
    pd.DataFrame(mlFeaturesLig).to_csv('{}mlFeaturesLig.csv'.format(targetFolder))

    if not os.path.isdir('{}alignedInteractionPharmacophores/'.format(targetFolder)):
        os.makedirs('{}alignedInteractionPharmacophores/'.format(targetFolder))
    if not os.path.isdir('{}alignedLigandPharmacophores/'.format(targetFolder)):
        os.makedirs('{}alignedLigandPharmacophores/'.format(targetFolder))

    for pdbCode, pInt, pLig in zip(pdbCodes, alignedPharmacophoresInt, alignedPharmacophoresLig):
        if pInt is not None:
            savePharmacophore(pInt, '{}alignedInteractionPharmacophores/{}.pml'.format(targetFolder, pdbCode))
        if pLig is not None:
            savePharmacophore(pLig, '{}alignedLigandPharmacophores/{}.pml'.format(targetFolder, pdbCode))

    print('Finished')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-targetFolder', required=True, type=str,
                        help='folder containing processed crystal structures of target. Used as output folder')
    parser.add_argument('-model', required=True, type=str, help='path to qphar model')
    args = parser.parse_args()

    main(args.targetFolder, args.model)
