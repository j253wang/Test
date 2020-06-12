import json
from azureml.core.runconfig import ContainerRegistry
from azureml.train.dnn import PyTorch
from Utils import read_from_json


def GetEstimator(environmentInfo, inpData):
    ''' Defines estimator for AML experiment.
        Method signature defined by AP.Data.

    Args:
        environmentInfo:
            workspace: The workspace with the correct svc that the run will be submitted to. This gives you access              to the default datastore, keyvault, container registry.
            datastore: Datastore where the data is located in
            compute : compute cluster the run should target
        inputData:
            dataDir: dataset directory
            dataset_name: name of dataset
            training_config: path of training configuration file
            toolsDir: directory with auxialiary tools
            scriptConfig: path of additional configuration for this method
            sourceDir: directory with experiment code
    '''

    conda_packages = None
    pip_packages = None

    # authenticated workspace. You can get access to the default key vault from here
    workspace = environmentInfo.workspace

    # https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.keyvault(class)?view=azure-ml-py
    keyvault = workspace.get_default_keyvault()

    ds = environmentInfo.datastore

    script_params = {
        "--data_dir": inpData.dataDir,
        "--training_config": json.dumps(read_from_json(inpData.training_config))
    }

    print(f'using script_params {script_params}')
    estimatorConfig = read_from_json(inpData.scriptConfig)

    conda_packages = estimatorConfig["conda_packages"]
    pip_packages = estimatorConfig["pip_packages"]
    print(
        f'got {conda_packages} and {pip_packages} from config for conda and pip packages')

    # default if json not past in or not set.
    if conda_packages is None:
        conda_packages = ["numpy", "pillow"]
    if pip_packages is None:
        pip_packages = ["facenet-pytorch",
                        "torch===1.4.0", "torchvision===0.5.0"]

    return PyTorch(
        source_directory=inpData.sourceDir,
        script_params=script_params,
        compute_target=environmentInfo.compute,
        entry_script='train_model_pytorch.py',
        use_gpu=True,
        source_directory_data_store=environmentInfo.datastore,
        conda_packages=conda_packages,
        pip_packages=pip_packages)
