#%%
import os
from os.path import isfile, join
from os import listdir
import csv

def make_dir_ifnot(direc):

    """A function that makes a directory if it hasn't existed"""

    if not os.path.exists(direc):
        os.makedirs(direc)
    print(direc, 'created')
    return 


def list_of_fullpath(folder, extension = '.png'):

    '''A function that returns a list of full path of the files in the input path'''

    file_list = [f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith(extension)]
    fullpath =[]
    for file in file_list:
        full = os.path.join(folder, file)
        fullpath.append(full)
    return fullpath

def list_of_filenames(folder, extension = '.png'):

    '''A function that returns a list of file names (without full path) in the input folder path'''

    file_list = [f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith(extension)]
    return file_list

#a = list_of_fullpath('E:/Physics year 3/BSc Project/Code/Yolov8/Greenhouse-cracks-11/valid/images/')
#print(a)

def rename_files(list_oldname, list_newname):
    for i in len(list_oldname):
        os.rename(list_oldname[i], list_newname[i])
    return 

def read_names_from_txt(file_path):
    """
    Read names from a txt file.
    
    Args:
    file_path (str): Path to the text file containing names.
    
    Returns:
    list: List of names read from the file.
    """
    with open(file_path, 'r') as file:
        names = file.readlines()
        # Remove any leading or trailing whitespace characters
        names = [name.strip() for name in names]
    return names

def save_csv(number_list, filename):
    '''A function that save a list of numbers into csv files '''
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Numbers"])
        for number in number_list:
            writer.writerow([number])
    return 'file saved'


def back_to_forward_slash(directory_path):
        '''A function that replaces backward slashes with forward ones in a string'''
        return directory_path.replace('\\', '/')



