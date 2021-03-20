from argparse import ArgumentParser
from utilities.modules import loadParams, loadMolecules, DEFAULT_TRAINING_PARAMETERS, splitSamplesActivities, saveMolecules, savePerformance, plotPredictionsFromMolecules
from utilities.modules import makeTrainingRun, makeTrainingTestRun
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-trainingSet', required=True, type=str, help='SDF-file containing training samples')
    parser.add_argument('-testSet', required=False, default=None, type=str, help='SDF-file containing _test samples')
    parser.add_argument('-p', required=False, type=str, default=None, help='path of parameters file')
    parser.add_argument('-o', required=True, type=str, help='folder where output files should be saved to')
    args = parser.parse_args()

    folder = args.o
    if not folder.endswith('/'):
        folder = '{}/'.format(folder)
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # load parameters
    params = DEFAULT_TRAINING_PARAMETERS
    if args.p is not None:
        customParams = loadParams(args.p)
        for key, value in customParams.items():
            params[key] = value
            if key == 'modelType':
                from utilities.modules import DEFAULT_MODEL_PARAMETERS

                params['modelParams'] = DEFAULT_MODEL_PARAMETERS[value]

    # load molecules
    trainingSet = loadMolecules(args.trainingSet)
    testSet = loadMolecules(args.testSet) if args.testSet is not None else None

    # split molecules / activities
    trainingMolecules, trainingActivities = splitSamplesActivities(trainingSet, 'activity')

    # train models
    if testSet is None:
        model, trainingPerformance, trainingMolecules = makeTrainingRun(trainingMolecules,
                                                                        trainingActivities,
                                                                        params)
        model.save('{}model/'.format(folder))
        saveMolecules(trainingMolecules, '{}trainingPredictions.sdf'.format(folder))
        savePerformance(trainingPerformance, '{}trainingPerformance'.format(folder))
        plotPredictionsFromMolecules(trainingSet, '{}training.png'.format(folder))

    else:  # testSet is not None
        testMolecules, testActivities = splitSamplesActivities(testSet, 'activity')
        model, trainingPerformance, trainingMolecules, testPerformance, testMolecules = makeTrainingTestRun(trainingMolecules,
                                                                                                            trainingActivities,
                                                                                                            testMolecules,
                                                                                                            testActivities,
                                                                                                            params)
        model.save('{}model/'.format(folder))
        saveMolecules(trainingMolecules, '{}trainingPredictions.sdf'.format(folder))
        savePerformance(trainingPerformance, '{}trainingPerformance'.format(folder))
        plotPredictionsFromMolecules(trainingSet, '{}training.png'.format(folder))
        saveMolecules(testMolecules, '{}testPredictions.sdf'.format(folder))
        savePerformance(testPerformance, '{}testPerformance'.format(folder))
        plotPredictionsFromMolecules(testSet, '{}test.png'.format(folder))
