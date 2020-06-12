
import glob
import random
import pandas as pd
import os
import numpy as np
import ntpath
import json
from collections import OrderedDict
import shutil

BOOLEAN_SCHEMA_TEMPLATE = "\n## {0}\n`bool`\n"
RECTFNULLABLE_SCHEMA_TEMPLATE = "\n## {0}\n`RectF?`\n"
FLOATNULLABLE_SCHEMA_TEMPLATE = "\n## {0}\n`float?`\n"

def read_input_meta_table(inputDir):
    '''
       Defines a method that will look within {inputDir} to find the first *.csv file
       This method assumes looking in a dedicated folder containing a single csv 

       Args:
            inputDir: Directory that contains the aptable as a csv.  
    '''
    print(f'looking in {inputDir} for csv table')
    tableFile = next((f for f in findFile(inputDir, "*.csv")), None)
    
    print(f'found {tableFile}')
    tableFile = os.path.join(tableFile)
    
    print(f'table file os path: {tableFile}')
    tableFileName = path_leaf(tableFile)

    print(f'parsing uuid from {tableFileName}')
    uuid = tableFileName.split(".")[0]

    print(f'found Uuid: {uuid}')

    print(f'looking for schema file .schema.md in {inputDir}')
    # look for schema file
    schemaFile = next((f for f in findFile(
        inputDir, "*.schema.md")), None)

    print(f'found schemaFile: {schemaFile}')
    schemaFile = os.path.join(schemaFile)

    # metadata table file is required.
    if not os.path.exists(tableFile):
        raise FileNotFoundError("Did not find table file")

    return tableFile, uuid, schemaFile

def write_output_schema(resultMetadataFolder, uuid,  schemaFile, columnDefinitions):
    '''
        Defines a method that will take the input {schemaFile}, copy it to {resultmetadataFolder}
        with name {uuid}.outputTable.metadata.schema.md. The schema.md is a constant, as the AP system will
        pick this up automatically when we load the table if the file is next to the table.
        Once the file is copied the {columnDefinitions} will be added to the new schema.

        Args: 
             resultMetadataFolder: directory to write schema file to.
             uuid: object ID to write into the file name
             schemaFile: originating schema file to append additional definitions to
             columnDefinitions: new definitions to add to the schema
    '''
    # copy the current schema of the table to the output result. This is where additional
    # type definitions can be specified to enable loading of typed tables within tooling.
    newSchema = os.path.join(resultMetadataFolder, f'{uuid}.outputTable.metadata.schema.md')

    print(f'copying {schemaFile} to {newSchema}')
    shutil.copy(schemaFile, newSchema)
    
    # append new typed columns to schema file
    with open(newSchema, 'a') as openFile:
        for col in columnDefinitions:
            print(f'adding {col} to schema')
            openFile.write(col)

def random_color(min, max):
    '''
        Defines a function that returns a random color. The colors RGB values 
        have constraint of {min} and {max}

        Args:
            min: Min value for RGB
            max: Max value for RGB
    '''
    return tuple([random.randint(min, max), random.randint(
        min, max), random.randint(min, max)])

def findFile(baseDir, searchString):
    '''
        Helper function for finding files within {baseDir} that match {searchString}
        for example within c:\\input\\ap_metadata find *.csv

        Args:
             baseDir: directory to look for the files within
             searchString: search pattern to use eg *.csv
    '''
    strPath = os.path.join(baseDir, searchString)
    print(f'looking in {baseDir} for {searchString}')
    return [f for f in glob.glob(strPath)]

def find_base_images(baseDir):
    '''
        Function used by PrepareDataset.py to find all png images for processing within {baseDir}
        Args:
             baseDir: Directory that contains the png files.
    '''
    return findFile(baseDir, "*.png")

def path_leaf(path):
    '''
        Helper function to get FileName from path
        Args:
             path: path to parse file name from
    '''
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def create_dir_if_not_Exist(path):
    '''
        Helper function to create a directory if it does not exist.
        Args: 
             path: directory to check and create
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def read_from_json(file_path, object_pairs_hook=OrderedDict, object_hook=None):
    '''
        Helper function to load data from a json file. Used to read config files.
        Args:
            file_path: json file to read
            
            object_hook: is an optional function that will be called with the
            result of any object literal decode (a ``dict``). The return value of
            ``object_hook`` will be used instead of the ``dict``. This feature
            can be used to implement custom decoders (e.g. JSON-RPC class hinting).
            
            object_pairs_hook:is an optional function that will be called with the
            result of any object literal decoded with an ordered list of pairs.  The
            return value of ``object_pairs_hook`` will be used instead of the ``dict``.
            This feature can be used to implement custom decoders that rely on the
            order that the key and value pairs are decoded (for example,
            collections.OrderedDict will remember the order of insertion). If
            ``object_hook`` is also defined, the ``object_pairs_hook`` takes priority.

            To use a custom ``JSONDecoder`` subclass, specify it with the ``cls``
            kwarg; otherwise ``JSONDecoder`` is used.
    '''
    
    with open(file_path, encoding='utf8') as json_file:
        data = json.load(
            json_file, object_pairs_hook=object_pairs_hook, object_hook=object_hook)
    return data
