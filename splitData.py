from argparse import ArgumentParser
from utils.modules import splitData, loadMolecules, saveMolecules
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', required=True, type=str, help='SDF-file containing input molecules')
    parser.add_argument('-a', required=True, type=str, help='name of activity property')
    parser.add_argument('-validationFraction', required=False, type=float, default=None,
                        help='fraction of validation data. Value between 0 and 1')
    parser.add_argument('-testFraction', required=False, type=float, default=None,
                        help='fraction of test data. Value between 0 and 1')
    parser.add_argument('-o', required=True, type=str, help='folder where output files should be saved to')
    args = parser.parse_args()

    # load molecules
    molecules = loadMolecules(args.i)

    # split data
    splits = splitData(molecules,
                       args.a,
                       validationFraction=args.validationFraction,
                       testFraction=args.testFraction
                       )

    # save data
    folder = args.o
    if not folder.endswith('/'):
        folder = '{}/'.format(folder)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    # mol_to_sdf(splits['trainingSet'], '{}trainingSet.sdf'.format(folder), multiconf=True)
    saveMolecules(splits['trainingSet'], '{}trainingSet.sdf'.format(folder))
    if len(splits['validationSet']) > 0:
        # mol_to_sdf(splits['validationSet'], '{}validationSet.sdf'.format(folder), multiconf=True)
        saveMolecules(splits['validationSet'], '{}validationSet.sdf'.format(folder))
    if len(splits['testSet']) > 0:
        # mol_to_sdf(splits['testSet'], '{}testSet.sdf'.format(folder), multiconf=True)
        saveMolecules(splits['testSet'], '{}testSet.sdf'.format(folder))
