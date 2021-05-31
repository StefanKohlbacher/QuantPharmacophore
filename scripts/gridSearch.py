from argparse import ArgumentParser
from src.modules import loadParams, loadMolecules, saveMolecules, savePerformance
from src.modules import gridSearch


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-trainingSet', required=True, type=str, help='SDF-file containing training samples')
    parser.add_argument('-validationSet', required=True, type=str, help='SDF-file containing validation samples')
    parser.add_argument('-testSet', required=True, default=None, type=str, help='SDF-file containing _test samples')
    parser.add_argument('-p', required=True, type=str, default=None, help='path of parameters file')
    parser.add_argument('-o', required=False, type=str, help='folder where output files should be saved to')
    parser.add_argument('-nrProcesses', required=False, type=int, default=1, help='number of processes to use in parallel')
    args = parser.parse_args()

    # load parameters
    searchParams = loadParams(args.p)

    # _test parameter combinations and select best models
    gridSearch({'trainingSet': args.trainingSet, 'validationSet': args.validationSet, 'testSet': args.testSet},
               searchParams,
               nrProcesses=args.nrProcesses,
               outputPath=args.o
               )
    print('Finished')
