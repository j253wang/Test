from Utils import create_dir_if_not_Exist, read_from_json, path_leaf, find_base_images, random_color, read_input_meta_table, write_output_schema, BOOLEAN_SCHEMA_TEMPLATE
from PIL import Image
import glob
import random
import pandas as pd
import os
import numpy as np
import ntpath
import getpass
import json
from joblib import Parallel, delayed
from collections import OrderedDict
from functools import reduce
import shutil


def find_images(inputDir, imageSampleCount):
    '''
        Function used by PrepareDataset.py to find all png images for processing within {baseDir}
        Args:
             baseDir: Directory that contains the png files.
             imageSampleCount: number of images to sample out of the ones found. This does not provide unique images,
             if the number is larger than the number of images found duplicates will get used.
    '''
    images = find_base_images(inputDir)

    # sample the found images to imageSampleCount. This API can produce duplicates
    images = random.choices(images, k=imageSampleCount)
    return images


def StartAggregate(inputData):
    ''' Defines method that will be run in ProcessClip stage on each
       individual clip.
       Method signature defined by AP.Data.

    Args:
        inputData:
            dataDir: Data which contains the dataset staged.
            toolsDir: directory with auxiliary tools (if any)
            scriptConfig: path of additional configuration for this method
            resultDir: directory where the resulting files are expected to be placed. This is what will be used in subsequent stages.
    '''
    # create result directory if not there
    create_dir_if_not_Exist(inputData.resultDir)

    # Script file is required throw if not founds
    if not os.path.exists(inputData.scriptConfig):
        raise FileNotFoundError(f"Did not find {inputData.scriptConfig}")

    settings = read_from_json(inputData.scriptConfig)

    # location of all images and associated metadata
    inputDir = inputData.dataDir

    # AP will aggregate and upload metadata files within AP_Metadata in the result folder
    # This will be used in training to output results metadata
    resultMetadataFolder = f'{inputData.resultDir}\\AP_Metadata'
    create_dir_if_not_Exist(resultMetadataFolder)

    # During ingestion an aptable was generated and saved. AP will save a CSV version (with associated schema) automatically
    # this file contains all metadata for the images
    tableFile, uuid, schemaFile = read_input_meta_table(
        inputDir)

    # any new column definitions go here.
    columnDefinitions = []
    write_output_schema(resultMetadataFolder, uuid,
                        schemaFile, columnDefinitions)

    # read the table file that contains metadata for each overlay we will work with
    df = pd.read_csv(tableFile, sep=',', header='infer')

    # grab the current columns and create an empty data frame
    # new data frame will contain the propagated metadata and new information about
    # the created image.
    cols = df.columns
    emptyDf = pd.DataFrame(columns=cols)


    # generate the images in parallel
    #results = Aggregation()
    results = []

    # flatten the results and iterate through them to insert them into the empty data frame
    # here we will decide if the images are part of a data Training or Testing data
    for  row in reduce(list.__add__, results):
        #add any additional metadata 

        # add row, and append new columns to it
        emptyDf = emptyDf.append(row, ignore_index=True, sort=False)

    # name as .metadata.csv inside ap_metadata folder to ensure it gets picked up by the handler
    # the schema file next to it will define the types for the data in this table.
    # keep the UUID in the table to avoid collisions accross tasks. ap handler will merge all of them in the next step into one
    # and propegate the schema file.
    emptyDf.to_csv(f'{resultMetadataFolder}\\{uuid}.outputTable.metadata.csv')
