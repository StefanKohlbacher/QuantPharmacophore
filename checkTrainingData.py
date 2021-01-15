from argparse import ArgumentParser
from utilities.modules import checkDataRequirements, loadMolecules


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', required=True, type=str, help='SDF-file containing input molecules')
    parser.add_argument('-a', required=True, type=str, help='name of activity property')
    args = parser.parse_args()

    # load molecules and extract activities
    molecules = loadMolecules(args.i)
    checkDataRequirements(molecules, activityName=args.a)
    print('Finished')
