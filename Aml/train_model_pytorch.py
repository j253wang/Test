import argparse
import pandas as pd
from functools import reduce
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import numpy as np
from azureml.core.run import Run
from facenet_pytorch import MTCNN, InceptionResnetV1,  extract_face
from PIL import Image, ImageDraw
from Utils import read_from_json, create_dir_if_not_Exist, read_input_meta_table, write_output_schema, BOOLEAN_SCHEMA_TEMPLATE, FLOATNULLABLE_SCHEMA_TEMPLATE, RECTFNULLABLE_SCHEMA_TEMPLATE
import json
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        help='directory with training data',
        required=True)
    parser.add_argument(
        '--training_config',
        type=str,
        help='Training config file',
        required=True)
    return parser.parse_args()


def run_training(images, inputdir, logSampleRage):

    preTrainedModel = os.path.join("pretrainedModel", "vggface2.pt")
    exists = os.path.exists(preTrainedModel)
    print(f'preTrainedModel {preTrainedModel} exists: {exists}')
    outputsModel =  os.path.join("outputs", "vggface2.pt")
    shutil.copy(preTrainedModel, outputsModel)
    run = Run.get_context()
    for i, row in images.iterrows():
        if np.random.rand() > logSampleRage:
            run.log("Image", i)
            run.log("fps", np.random.rand())
            run.log("accuracy", np.random.rand())
            run.log("confidence", np.random.rand())


def run_eval(images, inputdir, mtcnn, outputDir):
    '''
    Defines method will on the given images within input directory run
    through the provided mtcnn detector to run find a face on the image.
    The function will return the row from the original data frame table that has all metadata about the image
    along with information for the bounding box for the face detected (if any), the probability
    and the file generated by the mtcnn.

    Args:
        inputData:
            images: Array of the image names within inputdir to use
            inputdir: directory containing the images specified by images
            mtcnn: MTCNN to use to run face detection
            outputDir: directory to save output of MTCNN in.
    '''
    results = []
    for i, row in images.iterrows():
        imageName = row["NewImage"]
        imagePath = os.path.join(inputdir, imageName)
        img = Image.open(imagePath)
        outFile = f'detected_face_{imageName}'
        # Get cropped and prewhitened image tensor
        savepath = os.path.join(outputDir, outFile)
        img_cropped = mtcnn(img, save_path=savepath)
        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

        box = []
        x = None
        y = None
        width = None
        height = None
        if(boxes is not None):
            _box = boxes[0]
            x = _box[0]
            y = _box[1]
            width = _box[2] - x
            height = _box[3] - y

        detected = x is not None and y is not None and width is not None and height is not None

        row['FaceDetected'] = detected
        row['X'] = x
        row['Y'] = y
        row['Width'] = width
        row['Height'] = height
        row['confidence'] = probs[0]
        row['dtImageName'] = outFile
        landmark = None
        if landmarks is not None:
            landmark = landmarks[0]
            if landmark is not None:
                landmark = landmarks.tolist()

        row['landmarks'] = landmark
        results.append(row)
    return results

def main():
    ''' Main method '''
    args = parse_args()
    
    print(f'parse args: {args}')

    inputdir = os.path.join(args.data_dir)
    outputDir = os.path.join("outputs")
    inputTableDir = os.path.join(inputdir, 'AP_Metadata')
    resultMetadataFolder = os.path.join(outputDir, 'AP_Metadata')
    print(args.training_config)
    trainingConfig = args.training_config
    print(type(trainingConfig))
    jsonconfig = json.loads(trainingConfig)

    print(f'jsonConfig: {jsonconfig}')
    print(
        f'Starting training with inputdir:{inputdir}, writing results to {outputDir} and training config: {trainingConfig}.')
    print(
        f'Will read AP Metadata table from {inputTableDir} and write a new one to {resultMetadataFolder}')
    tableFile, uuid, schemaFile = read_input_meta_table(inputTableDir)

    if(tableFile == None):
        raise Exception("Did not find table file " + tableFile)
    sampleStr = jsonconfig['SampleRate']
    print(f'sampleStr {sampleStr}')

    logSampleRage = float(sampleStr)
    # read the table file that contains metadata for each overlay we will work with
    create_dir_if_not_Exist(resultMetadataFolder)
               
    colDefs = [FLOATNULLABLE_SCHEMA_TEMPLATE.format('FaceDetected'),
               FLOATNULLABLE_SCHEMA_TEMPLATE.format('Confidence'),
               FLOATNULLABLE_SCHEMA_TEMPLATE.format('X'),
               FLOATNULLABLE_SCHEMA_TEMPLATE.format('Y'),
               FLOATNULLABLE_SCHEMA_TEMPLATE.format('Width'),
               FLOATNULLABLE_SCHEMA_TEMPLATE.format('Height')]

    write_output_schema(resultMetadataFolder, uuid, schemaFile, colDefs)
    df = pd.read_csv(tableFile, sep=',', header='infer')

    # grab the current columns and create an empty data frame
    # new data frame will contain the propagated metadata and new information about
    # the created image.
    cols = df.columns.tolist()
    emptyDf = pd.DataFrame(columns=cols)

    # filter to images used within training
    trainImages = df.loc[df["IsTrainData"] == True]

    # filter to all images that are train Data and Validation
    valImages = df.loc[(df["IsTrainData"] == True) & (df["IsValData"] == True)]

    # filter to all test images
    testImages = df.loc[df["IsTestData"] == True]

    # Run training and model generation
    run_training(trainImages, inputdir, logSampleRage)

    # instantiate MTCCN, using pretrained models here
    # you would consume model generated by stage above here.
    mtcnn = MTCNN(image_size=512, margin=512, post_process=False)
    
    rez = []

    # run evaluation on validation images
    rez.append(run_eval(valImages, inputdir, mtcnn, outputDir))

    # run evaluation on test images
    rez.append(run_eval(testImages, inputdir, mtcnn, outputDir))

    # merge the results and add them into a new data frame table
    for num, row in enumerate(reduce(list.__add__, rez), start=0):
        emptyDf = emptyDf.append(row, ignore_index=True, sort=False)

    csvPath = os.path.join(resultMetadataFolder,
                           f'{uuid}.outputTable.metadata.csv')
    emptyDf.to_csv(csvPath, index=False)

if __name__ == '__main__':
    main()