import sys
sys.path.append('../Scripts/GithubRepo/')
from trainingValidationTest import main
import multiprocessing as mp
import os
import json


def run_parallel(nr_processes, jobs):
    # set up some overhead
    processes = {}  # keep track of all the processes we started
    scheduledJobs = mp.Queue()
    for job in jobs:
        scheduledJobs.put(job)
    finishedJobs = mp.Queue()  # subprocesses fill this queue with process id once finished

    def run(args, process_name):
        main(args)
        finishedJobs.put(process_name)

    # spawn processes
    for i in range(nr_processes):
        p = mp.Process(target=run, args=(scheduledJobs.get(), i), name=str(i))
        processes[i] = p
        p.start()  # immediately starts running the process

    # kill and restart processes once done to clean up memory and start fresh
    running_processes = len(processes)
    while running_processes > 0:
        i = finishedJobs.get()  # blocks until we get a new entry
        p = processes.pop(i)
        p.terminate()

        # check whether there is something left to do -> start over
        if not scheduledJobs.empty():
            p = mp.Process(target=run, args=(scheduledJobs.get(), i), name=str(i))
            processes[i] = p
            p.start()

        running_processes = len(processes)

    # all jobs ran successfully -> clean up
    scheduledJobs.close()
    finishedJobs.close()
    for i in processes.keys():  # should be empty already, but better check
        p = processes.pop(i)
        p.terminate()


if __name__ == '__main__':
    # define parameterss
    nrProcesses = 1  # !!! be aware of potential memory leak!
    basePath = '../Data/Evaluation_datasets/'
    params = {
        'basePath': basePath,
        'inputFile': 'conformations.sdf',
        'o': 'gaba_a_case_study',
        'activityName': 'pchembl_value',
        'fuzzy': True,
        # 'threshold': 1.5,  # 1.5 is radius of pharmacophores
        # 'mostRigidTemplate': True,  # True per default
    }
    searchParams = {  # only parameters to be combined!
        'weightType': ['distance', 'nrOfFeatures', None],
        'modelType': ['randomForest', 'pca_ridge', 'ridge', 'pls', 'pca_lr'],
        'threshold': [1, 1.5, 2]
    }
    modelParams = {
        'ridge': {'fit_intercept': False},
        # 'lasso': {'fit_intercept': False},
        'randomForest': {'n_estimators': 10, 'max_depth': 3},
        'pca_ridge': {},
        # 'pca_lasso': {},
        'pls': {},
        'pca_lr': {},
    }

    # get targets
    # with open('{basePath}evaluation_targets.json'.format(basePath=basePath), 'r') as f:
    #     targetDict = json.load(f)
    targetDict = {
        'GABA_A_a1_b3_g2': {'assayID': ['CHEMBL676826', 'CHEMBL678329', 'CHEMBL822162']},
        'GABA_A_a2_b3_g2': {'assayID': ['CHEMBL1273616', 'CHEMBL681841']},
        'GABA_A_a3_b3_g2': {'assayID': ['CHEMBL678741', 'CHEMBL680432', 'CHEMBL823761']},
        'GABA_A_a5_b3_g2': {'assayID': ['CHEMBL678745', 'CHEMBL685337', 'CHEMBL685338']},
        'GABA_A_a6_b3_g2': {'assayID': ['CHEMBL683045']},
    }
    nrAssaysTotal = 1
    for target, information in targetDict.items():
        nrAssaysTotal *= len(information['assayID'])

    # versioning
    i = 1
    while os.path.isdir('{f}_{i}/'.format(f=params['o'], i=str(i))):
        i += 1
    filename = '{f}_{i}/'.format(f=params['o'], i=str(i))
    os.mkdir(filename)

    jobs = []  # store all the information passed in a process to the main() function here in form of a tuple
    # iterate over targets
    for name, information in targetDict.items():
        if not isinstance(information['assayID'], list):
            information['assayID'] = [information['assayID']]

        # iterare over assays
        for assay in information['assayID']:
            tempParams = {
                'dataset': '{target}/{assay}/'.format(target=name, assay=assay),
                'name': '{target}_{assay}'.format(target=name, assay=assay),
                'logPath': '{path}{folder}/{assay}/'.format(path=filename, folder=name, assay=assay),
            }

            # check whether has already been processed --> just in case we have to rerun due to some error, so we do not unnecessary waste resources
            # if os.path.isfile('{logPath}finished.log'.format(logPath=tempParams['logPath'])):  # already processed successfully
            #     continue

            for key, value in params.items():
                tempParams[key] = value

            # load data split
            with open('{basePath}{folder}{cvFolds}.json'.format(basePath=basePath,
                                                                folder=tempParams['dataset'],
                                                                cvFolds='trainValidationTestSplit'), 'r') as f:
                trainValTestSplit = json.load(f)

            # apparently we do not have any test samples -> skip target
            if trainValTestSplit['test'] == 0 or trainValTestSplit['training'] == 0 or trainValTestSplit['validation'] == 0:
                continue

            jobs.append((tempParams, trainValTestSplit, searchParams, modelParams))

    run_parallel(nrProcesses, jobs)
    print('Finished training-validation-test run')
