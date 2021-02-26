import os
import cv2
import h5py
import numpy as np
import configparser
from skimage.feature import hog

__FE__ = ['hog', 'phog']
__PP__ = ['crop', 'gray']

__cfg__ = configparser.RawConfigParser()

def get_image_labels_and_paths(path_dir):
    unique_labels = [] #contain unique label
    all_labels = [] #contain all lebels, allow duplicate
    all_image_paths = []
    
    label_dir = sorted(os.listdir(path_dir))
    for label in label_dir:
        unique_labels.append(label)
        images = os.listdir(f"{path_dir}/{label}")
        paths = []
        labels = []
        print(f"import {len(images)} files from {path_dir}/{label}")
        for image in images:
            image_path = f"{path_dir}/{label}/{image}"
            paths.append(image_path)
            labels.append(label)

        all_image_paths.append(paths)
        all_labels.append(labels)
    
    return (unique_labels, all_labels, all_image_paths)

def extract_phog(image, orientation=9, pixel_per_cell=(8, 8), cell_per_block=(2, 2), is_visualize=True, is_multichannel=True, level=0):
    grid_x = pow(2, level)
    grid_y = grid_x

    h, w, t = image.shape
    tile_col_size = round(w / grid_x)
    tile_row_size = round(h / grid_y)

    # tile_hog_images = []
    tile_hog_desc_concat = []
    for r in range(0, h, tile_row_size):
        for c in range(0, w, tile_col_size):
            tile_image = image[r:r+tile_row_size, c:c+tile_col_size,:]
            hog_desc, hog_image = hog(tile_image, orientations=orientation, pixels_per_cell=pixel_per_cell, cells_per_block=cell_per_block, visualize=is_visualize, multichannel=is_multichannel)
            # tile_hog_images.append(hog_image)
            tile_hog_desc_concat = np.concatenate((tile_hog_desc_concat, hog_desc))        

    return tile_hog_desc_concat

def apply(cfg_file_name):
    dt = 'data'
    fe = 'phog'
    pp = 'crop'
    __cfg__.read(cfg_file_name)
    __dict_data__ = dict(__cfg__.items(dt))
    __dict_crop__ = dict(__cfg__.items(pp))
    __dict_phog__ = dict(__cfg__.items(fe))

    print(__dict_data__)
    location = __dict_data__['path']
    location_section = location.split('/')
    purpose = f"{location_section[len(location_section)-1]}ing"

    print(__dict_crop__)
    target_width = int(__dict_crop__['target-width'])
    target_height = int(__dict_crop__['target-height'])
    
    print(__dict_phog__)
    lv = int(__dict_phog__['level'])
    nbin = int(__dict_phog__['bin'])
    ppc = eval(__dict_phog__['pixels-per-cell'])
    cpb= eval(__dict_phog__['cells-per-block'])
    visual = __dict_phog__['visualize'] == 'True'
    multichan = __dict_phog__['multichannel'] == 'True'
    
    print("\n[Summary]")
    print(f"Purpose     : {purpose}")
    print(f"Location    : {location}")
    print(f"PP Method   : {pp} {target_width}x{target_height}")
    print(f"FE Method   : {fe} Lv {lv}, bin {nbin}, pixel per cell {ppc}, cell per block {cpb}")

    print("\n\n")
    print("=== Pre-Processing and Feature Extraction ===")
    print("===                START                  ===")

    # [Import] import raw images
    print("[Import] importing images ...")
    unique_labels, all_labels, all_image_paths = get_image_labels_and_paths(location)
    print("[Import] import done ...")

    # [PPFE] apply pre-processing and feature extraction
    y = 0
    x = 0

    h5py_file_name = f"{dt}_{purpose}_{fe}_{lv}.h5"

    hf = h5py.File(h5py_file_name, 'w')
    groups = []
    for label in unique_labels:
        group = hf.create_group(label)
        groups.append(group)

    for group_index in range(0, len(groups)):
        group_name = unique_labels[group_index]
        group = groups[group_index]
        paths = all_image_paths[group_index]
        for path_index in range(0, len(paths)):
            path = paths[path_index]
            print(f"Processing image of label({group_name})[{path_index}]")
            # [Read] read image from path
            image_file = cv2.imread(path)

            # [Crop] crop image
            image_file = image_file[y:y+target_height, x:x+target_width]

            # [PHOG] implement 
            hog_desc = extract_phog(
                image=image_file, 
                orientation=nbin,
                pixel_per_cell=ppc,
                cell_per_block=cpb, 
                is_visualize=visual,
                is_multichannel=multichan,
                level=lv
            )

            # [Save] save result into file
            group.create_dataset(f"{group_name}{path_index}", data=hog_desc)
        
    hf.close()
    print("===                DONE                   ===")


    hf = h5py.File(h5py_file_name, 'r')
    for key in hf.keys():
        group = hf.get(key)
        print(group.items())

def executes():
    print("=== Pre-Processing and Feature Extraction ===")
    print("[Purpose]")
    print(f"1. [train] Training Purpose")
    print(f"2. [test ] Testing Purpose")
    print("Set purpose of this PPFE by type the word inside [] without any space.")
    purpose = input("Purpose: ")

    print("[Import]")
    print("Set dataset source location.")
    location = input("Location: ")

    print("\n[Pre-Processing]")
    print(f"1. [{__PP__[0]} ] Crop")
    print(f"2. [{__PP__[1]} ] Grayscale (Coming Soon)")
    print("Choose pre-processing method by type the word inside [] without any space.")
    pp = input("Method: ")

    target_width = 0
    target_height = 0
    if (pp == __PP__[0]):
        print("Define target width and height in integer.")
        target_width = int(input("Width : "))
        target_height = int(input("Height: "))
    else:
        print("Sorry, method is not available yet")
        # Coming soon
        target_width = 0
        target_height = 0

    print("\n[Feature Extraction]")
    print(f"1. [{__FE__[0]} ] Histogram of Gradient")
    print(f"2. [{__FE__[1]}] Pyramidal Histogram of Gradient")
    print("Choose feature extraction method by type the word inside [] without any space.")
    fe = input("Method: ")

    lv = 0
    if (fe == __FE__[1]):
        print("Define phog level in integer.")
        lv = int(input("Level: "))

    print("\n[Summary]")
    print(f"Purpose     : {purpose}")
    print(f"Location    : {location}")

    if (pp == __PP__[0]):
        print(f"PP Method   : {pp} {target_width}x{target_height}")
    else:
        print(f"FP Method   : {pp}")
    
    if (fe == __FE__[1]):
        print(f"FE Method   : {fe} Lv {lv}")
    else : 
        print(f"FE Method   : {fe}")


    print("\n\n")
    print("=== Pre-Processing and Feature Extraction ===")
    print("===                START                  ===")

    # [Import] import raw images
    print("[Import] importing images ...")
    unique_labels, all_labels, all_image_paths = get_image_labels_and_paths(location)
    print("[Import] import done ...")

    # [PPFE] apply pre-processing and feature extraction
    y = 0
    x = 0

    h5py_file_name = f"{purpose}_data_{fe}_lv{lv}.h5"

    hf = h5py.File(h5py_file_name, 'w')
    groups = []
    for label in unique_labels:
        group = hf.create_group(label)
        groups.append(group)

    for group_index in range(0, len(groups)):
        group_name = unique_labels[group_index]
        group = groups[group_index]
        paths = all_image_paths[group_index]
        for path_index in range(0, len(paths)):
            path = paths[path_index]
            print(f"Processing image of label({group_name})[{path_index}]")
            # [Read] read image from path
            image_file = cv2.imread(path)

            # [Crop] crop image
            image_file = image_file[y:y+target_height, x:x+target_width]

            # [PHOG] implement 
            hog_desc = extract_phog(image_file, level=lv)

            # [Save] save result into file
            group.create_dataset(f"{group_name}{path_index}", data=hog_desc)
        
    hf.close()
    print("===                DONE                   ===")


    hf = h5py.File(h5py_file_name, 'r')
    for key in hf.keys():
        group = hf.get(key)
        print(group.items())
