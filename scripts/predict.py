from argparse import ArgumentParser
import os
import json
from src.modules import loadMolecules, saveMolecules, addPropertyToSDFData
from src.pharmacophore_tools import loadPharmacophore
from src.hyperpharmacophore import DistanceHyperpharmacophore


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', type=str, required=True, help='path to SDF-file, folder containing PML-files or single PML-file')
    parser.add_argument('-m', type=str, required=True, help='path to model folder')
    parser.add_argument('-o', type=str, required=True,
                        help='path of output file. Either SDF for molecules or JSON for pharmacophores')
    args = parser.parse_args()

    # load model
    model = DistanceHyperpharmacophore()
    model.load(args.m)

    # load samples
    if args.i.endswith('sdf'):  # molecules
        # load molecules and predict batch
        molecules = loadMolecules(args.i)
        predictions, scores = model.predict(molecules, returnScores=True)

        # save molecules
        for mol, y_pred, score in zip(molecules, predictions, scores):
            addPropertyToSDFData(mol, 'prediction', y_pred)
            addPropertyToSDFData(mol, 'alignmentScore', score)
        saveMolecules(molecules, args.o)

    elif os.path.isdir(args.i):  # folder containing pml-files
        # check files and list pharmacophore files
        files = [f for f in os.listdir(args.i) if f.endswith('pml')]
        path = args.i if args.i.endswith('/') else '{}/'.format(args.i)

        samples = {}
        for f in files:
            # load pharmacophore and predict
            pharmacophore = loadPharmacophore('{}{}'.format(path, f))
            predictions, scores = model.predict(pharmacophore, returnScores=True)

            # store predictions
            filename = ''.join(f.split('.')[:-1])
            samples[filename] = {'prediction': predictions[0], 'alignmentScore': scores[0]}

        # save predictions
        with open(args.o, 'w') as f:
            json.dump(samples, f, indent=2)

    elif args.i.endswith('pml'):  # single pharmacophore file
        # load pharmacophore and predict single sample
        pharmacophore = loadPharmacophore(args.i)
        predictions, scores = model.predict(pharmacophore, returnScores=True)

        # save
        filename = ''.join(args.i.split('/')[-1].split('.')[:-1])
        with open(args.o, 'w') as f:
            json.dump({filename: {'prediction': predictions[0], 'alignmentScore': scores[0]}}, f, indent=2)

    else:
        print('Path not recognized. Please provide an SDF-file, a PML-file or a folder containing PML-files.')
        print(args.i, 'was given')
