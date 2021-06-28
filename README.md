# QuantPharmacophore
Code available for the quantitative pharmacophores  
<!-- Data for reproducing is available for download at:  https://drive.google.com/drive/folders/1kicJIM6hT0VAc-Ir83ex49YscUibP_GC?usp=sharing -->


## Setup
- Download and install docker (https://docs.docker.com/engine/install/). Make sure virtualization is enabled on your computer. 
Check this link on how to enable it -> "https://2nwiki.2n.cz/pages/viewpage.action?pageId=75202968" on how to enable virtualization on your machine.
- create and run docker container
```shell script
bash setup.sh
```

The above command will create a shared folder `~/container_data` on your host system with the docker container (`/qphar/data`). 
This will synchronise any files in these folders automatically and allows sharing of data as well as extracting data after running the
models. If you want to copy files manually, you can do so with the following commands.  
```shell script
docker cp <SOURCE_PATH_AT_HOST> <NAME_OF_YOUR_CONTAINER>:<TARGET_PATH_IN_CONTAINER>  # copy data into the container
docker cp <NAME_OF_YOUR_CONTAINER>:<TARGET_PATH_IN_CONTAINER> <SOURCE_PATH_AT_HOST>  # retrieve results or similar from container to host
```  

Start container: Execute this command if you have created the container already. (get a list of containers with 'docker ps -a') 
```shell script
docker start <NAME_OF_YOUR_CONTAINER>
docker attach <NAME_OF_YOUR_CONTAINER>
``` 

Once you are attached to a shell inside the container, you can run the scripts (see below). If you are sharing data with the container
make sure to provide that path for the scripts.  
## Usage
First, make sure to copy all the necessary data to your folder. This could be an SDF-file (with 3D conformations!) or a 
JSON-file specifying parameters. If not, default parameters are being used. For testing purposes, sample data is provided 
within the container as a subfolder to the working directory --> ./test/  
All examples below assume you mounted the volume '/data/' (see running image above), and all data will be taken and saved from there.
Also it assumes you attached the running contain to your shell and run all scripts from the working directory (/home/qphar). 
The files are the same as in the test-directory. 
The container provides five different functionalities: 
- check whether data fulfills all necessary requirements --> checkTrainingData.py
- split data into training, validation, test set --> splitData.py
- train a model with specified parameters on a given dataset --> train.py
- perform grid-search over a specified set of parameters on a given dataset --> gridsearch.py
- predict the activity of samples with a previously trained model --> predict.py

### check training data
Check whether your data is appropriate to use for 3D-QSAR. The following criteria are checked: 
- do the samples, molecules, have conformations?
- are activity values provided and are they a number?
- does the activity span at least 3 log units, or at least 2 log units not considering outliers?
- is the activity data somehow clustered or homogeneously distributed?

Execute the file 'checkTrainingData.py'. The parameters '-i', input SDF-file, and '-a', name of activity property, are required
```shell script
python checkTrainingData.py -i /data/molecules.sdf -a pchembl_value
``` 
This command will output that conformations are missing for all molecules in the file. Generate conformations and provide the new file. 
```shell script
python checkTrainingData.py -i /data/conformations.sdf -a pchembl_value
``` 
Running it again with conformations will prompt that the data fulfills all requirements to be used for 3D-QSAR.

### split data
Will split the data into training and optionally validation and/or test set. Required parameters are '-i' and '-a' 
(see checkTrainingData for explanation). Additionally, the parameter '-o' is required, which requires you to specfiy 
a folder where all the output data should be saved to. The parameters '-validationFraction' and '-testFraction' are
optional and allows you to specify the size of training, validation and test sets. Data is split first by test-fraction
and the remaining training-fraction is split into training-fraction and validation-fraction. If only test-fraction or
validation-fraction are given, then only the first step is carried out.  
i.e.: 100 samples. validationFraction=0.1, testFraction=0.2 --> testSet = 20 samples; validationSet = 8 samples; 
trainingSet = 72 samples
```shell script
mkdir /data/datasets
python splitData.py -i /data/conformations.sdf -a pchembl_value -o /data/datasets -validationFraction 0.1 -testFraction 0.2
``` 
This will generated the files 'trainingSet.sdf', 'validationSet.sdf' and 'testSet.sdf' in the directory '/data/datasets/'. 
These files are meant to be the input for the following scripts. 

### train 
Simply trains a model from a given trainingSet with optionally specified parameters. If a testSet is provided the model
will be evaluated on that dataset. Required parameters are '-trainingSet' and '-o', which are path to trainingSet and output folder. 
Parameters '-testSet' and '-p' are optional and specify path to testSet and parameters-file (JSON-format).

```shell script
mkdir /data/output
python train.py -trainingSet /data/datasets/trainingSet.sdf -testSet /data/datasets/testSet.sdf -o /data/output -p /data/trainingParams.json
```  
The parameters file might look like this: 
```json
{
  "fuzzy": true,
  "weightType": "distance",
  "logPath": "./log/",
  "modelType": "randomForest",
  "threshold": 1.5,
  "modelParams":  {"n_estimators":  10, "max_depth":  3},
  "mergeOnce": true,
  "metric": "R2"
}
```
Predictions for the trainingSet and testSet are saved as property to the SDF-file with property-name 'prediction'.
Additionally, a plot of true and predicted activity values is made and saved. Finally, the trained model is saved in the output
folder. 

### grid search
Since the optimal parameters for the model are not known beforehand, this is the file you would want to execute
after checking and splitting your data. First, make a file in JSON-format containing the parameters to be investigated in a list. 
Here we will name the file 'searchParams.json' and it looks like this: 
```json
{
  "weightType": ["distance", "nrOfFeatures", null],
  "modelType": ["randomForest", "ridge", "pca_ridge", "pca_lr"],
  "threshold": [1, 1.5, 2], 
  "metric": ["RMSE", "R2"]
}
```
We recommend to try variations in the parameters 'weightType', 'modelType', 'threshold' and possibly 'metric'. 
The 'gridSearch.py'-file requires the following parameters: '-trainingSet', '-validationSet', '-testSet' and '-p' 
(parameters file). The output folder '-o' is optional, whereas the default output is the working directory -> creates 
a folder named 'gridSearch_0' if not provided. The parameter '-nrProcesses' specifies the number of jobs to run 
in parallel, per default all jobs are run sequentially without parallelization. 

```shell script
python gridSearch.py -trainingSet /data/datasets/trainingSet.sdf -testSet /data/datasets/testSet.sdf -validationSet /data/datasets/validationSet.sdf -p /data/searchParams.json -o /data/gridSearch -nrProcesses 8
``` 
This will search all the parameter combinations, train models for all these combinations, evaluate the models on the validation set
and finally choose the best model on the test set. All intermediate models, their predictions on all datasets, as well as corresponding
plots are saved in the output folder. For each job testing one parameter combination a separate folder is created containing 
all the results. In that folder a file 'params.json' is also saved, to identify the parameters of the best model afterwards. 
Evaluated performance for each model is stored individually in these folders. The test results are also aggregated and saved in the 
output-folder.   
Note: the best model does not have to be retrained with the best set of parameters, since it was already saved. Identifying 
the job number by checking the results file, gets you the best model and all corresponding activity plots and predictions.  

### predict  
Simply loads the provided samples and a trained model and predicts the activity values of the given samples. 
Data is stored to the output folder. 
```shell script
python predict.py -i /data/datasets/testSet.sdf -m /data/gridSearch_0/0/model/ -o /data/datasets/predictions.sdf
``` 
If a folder is provided instead of an SDF-file, the folder is searched for all PML-files (pharmacophores). Predictions
are made for all pharmacophores and saved in a JSON-file, whereas the filename is associated with the predicted activity. 
```shell script
python predict.py -i /data/datasets/pharmacophores/ -m /data/gridSearch_0/0/model/ -o /data/datasets/predictions.json
``` 
Will yield a JSON-file looking something like: 
```json
{
  "pharmacophore_1.pml": {"prediction":  5.45, "alignmentScore":  0.45}, 
  "pharmacophore_2.pml": {"prediction":  7.15, "alignmentScore":  0.36},
  "pharmacophore_3.pml": {"prediction":  8.37, "alignmentScore":  0.89} 
}
```
