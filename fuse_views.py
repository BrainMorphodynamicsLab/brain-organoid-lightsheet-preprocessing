# Fuse views
import sys
import os
import shutil
import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from scipy import fftpack
import numpy
import tifffile
import glob
from shutil import copyfile
import uuid
from scipy.signal import medfilt
from tkinter import filedialog as FileDialog
from datetime import datetime

import logging
logging.getLogger('tifffile').setLevel(logging.ERROR)

cache_mask = None
cache_width = None
cache_height = None
cache_r_psf_pxl = None

criterion_threshold = 0.98

def dct_generate_mask(width, heigth, r_psf_pxl):
    global cache_width
    global cache_height
    global cache_mask
    global cache_r_psf_pxl

    if (cache_width == width and cache_height == heigth and cache_r_psf_pxl == r_psf_pxl):
        return cache_mask

    mask = numpy.zeros(shape=(heigth,width))
    for x in range(width):
        for y in range(heigth):
            if x+y < int(width/r_psf_pxl):
                mask[y][x] = True
            else: mask[y][x] = False

    cache_width = width
    cache_height = heigth
    cache_r_psf_pxl = r_psf_pxl
    cache_mask = mask

    return mask

def dct_fast(image, mask, r_psf_pxl, bin_factor):
    image = _rebin(image, bin_factor)
    width,_ = numpy.shape(image)

    Fc = fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')
    Fc_masked = numpy.multiply(Fc**2, mask)
    L2 = numpy.sum(Fc_masked)
    L2 = numpy.sqrt(L2)
    FL = Fc/L2
    
    DCTentr = -(2/(width/r_psf_pxl)**2)*numpy.sum(numpy.abs(FL[FL!=0])*(numpy.log2(numpy.abs(FL[FL!=0]))))

    return DCTentr

def _rebin(array, binning_factor):
    # get dimensions of original array
    shape = numpy.shape(array)
    binning_factor = int(binning_factor)
    
    # create square image if necessary
    if shape[0] != shape[1]:
        min_dim = numpy.amin(shape)
        array = array[0:min_dim, 0:min_dim]

    # get new dimensions of array
    shape = numpy.shape(array)
    
    # get new dimensions and potential left over
    new_shape = (int(shape[0]//binning_factor), int(shape[1]//binning_factor))
    leftover = (int(shape[0]%binning_factor), int(shape[1]%binning_factor))

    # remove leftover(s) at the right end of the array
    if leftover[0] != 0 or leftover[1] != 0:
        array = array[0 : -leftover[0], 0 : -leftover[1]]

    # tupple for reshapping
    shape = (new_shape[0], binning_factor, new_shape[1], binning_factor)
    
    return array.reshape(shape).mean(-1).mean(1)

def get_binned_pixel(pixel_size):
    # microscope parameters 
    # compute rayleigh radius of the microscope
    NA = 1.1

    wavelength = 0.525
    reality_factor = 1.5 # degradation of the PSF
    r_psf = reality_factor*(0.61*wavelength/NA)

    # compare to pixel size to determine binning factor
    r_psf_pxl = r_psf/pixel_size
    bin_factor = r_psf_pxl//0.5
    r_psf_pxl_bin = r_psf_pxl / bin_factor

    return r_psf_pxl_bin, bin_factor

def update_parameter(line, to_find, length, new_value):
    size_start = line.find(to_find)
    if size_start != -1:
        size_end = line.find('"', size_start + length)
        if isinstance(new_value,str):
            old_line = line[size_start:size_end]
            new_line = new_value
        else:
            old_line = line[size_start:size_end+1]
            new_line = to_find + str(new_value) + '"'
        line = line.replace(old_line, new_line)
        
    return line

def creating_directory(path, allow_existing=False):
    """Create an output directory while preserving the legacy fail-if-present default."""
    if os.path.exists(path):
        if allow_existing:
            return
        print('\nAn output folder already exists \nOperation stopped.')
        raise SystemExit(1)
    try:
        os.makedirs(path)
    except OSError:
        print("\nCreation of the directory %s failed" % path)
        raise SystemExit(1)

def annotate_report(crop, file_name_view1, saving_path, view_change_index):
    position_time_start = file_name_view1.find('/')
    position_time_end = file_name_view1.find('.tif')
    position_time_name = file_name_view1[position_time_start+1:position_time_end-6]
    report_name = 'Switching plane report.txt'
    report = open(os.path.join(saving_path,report_name), 'a')
    with open(os.path.join(saving_path,report_name), 'r') as report_to_read:
        if len(report_to_read.readlines())==0:
            # report.write('Threshold = ' + str(criterion_threshold) + '\n')
            report.write('Crop size (microns): ' + str(crop) + '\n')
            report.write('Switching plane corresponds to the last plane of View1\n')
    report.write(position_time_name + '\t' + str(view_change_index) + '\n')
    report.close()

class Script:
    def __init__(self):
        # list of unique parameter identifiers
        self.parameters = [
            "input", "output_root", "settings_to_fuse", "selected_positions",
            "view_1_token", "view_2_token", "fused_token", "ome_glob_pattern",
            "reference_channel","crop",
            "tp_start", "tp_end",
            "save_as_crop", "display_graphs"
        ]

        # label of each parameter in user interface
        self.parameter_label = {
            "input": "Input folder path",
            "output_root": "Output folder (or <auto>)",
            "settings_to_fuse": "Settings to fuse",
            "selected_positions": "Selected positions (ALL or comma-separated)",
            "view_1_token": "View 1 token",
            "view_2_token": "View 2 token",
            "fused_token": "Fused token",
            "ome_glob_pattern": "OME companion glob pattern",
            "reference_channel": "Reference channel for fusion",
            "crop": "Crop [um]",
            "save_as_crop": "Save cropped image",
            "tp_start": "First timepoint",
            "tp_end": "Last timepoint",
            "display_graphs": "Display graphs"
        }

        # default parameters values loaded to user interface
        self.parameter_default_value = {
            "input": "C:/",
            "output_root": "<auto>",
            "settings_to_fuse": "Settings 1",
            "selected_positions": "ALL",
            "view_1_token": "View1",
            "view_2_token": "View2",
            "fused_token": "Fused",
            "ome_glob_pattern": "*.ome",
            "reference_channel": "",
            "crop": 950,
            "save_as_crop": True,
            "tp_start": 1,
            "tp_end": 10000,
            "display_graphs": False
        }

        # types of parameters, min and max values provided in tuple
        self.parameter_type = {
            "input": str,
            "output_root": str,
            "settings_to_fuse": str,
            "selected_positions": str,
            "view_1_token": str,
            "view_2_token": str,
            "fused_token": str,
            "ome_glob_pattern": str,
            "reference_channel": str,
            "crop": (int, 50, 950),
            "save_as_crop": bool,
            "tp_start": int,
            "tp_end": int,
            "display_graphs": bool
        }

        # control to use in user interface, if not defined text box will be used
        self.parameter_control = {
            "input": "folder",
            "output_root": "entry",
            "save_as_crop": "checkbox",
            "display_graphs": "checkbox"
        }

        # documentation shown in user interface for each parameter
        self.parameter_help = {
            "input": "Path to the folder to analyze.",
            "output_root": "Use <auto> to preserve the original sibling <input>_fused behavior, or supply an explicit output folder.",
            "settings_to_fuse": "Settings to fuse.",
            "selected_positions": "ALL preserves the interactive position picker. A comma-separated list skips the picker.",
            "view_1_token": "Filename/channel token identifying the first camera view.",
            "view_2_token": "Filename/channel token identifying the second camera view.",
            "fused_token": "Token written into fused filenames and OME references.",
            "ome_glob_pattern": "Glob used to find the single OME companion in each settings folder.",
            "reference_channel": "Channel used to determine the switching plane. The plane is then applied to the other channels.",
            "crop": "Square crop taken at the center of the image and used to determine the switching plane. Must be between 50 and 950. If larger than the image size, the whole image is used",
            "save_as_crop": "If (un-)ticked, fused images are saved (un-)cropped",
            "tp_start": "Timepoint from which to start fusing",
            "tp_end": "Timepoint at which to stop fusing",
            "display_graphs": "If ticked, calibration graphs for fusion will be displayed",
        }

        self.window_tittle = "Fuse views"

        self.view1_scores = []
        self.view2_scores = []
        self.view1_scores_hat = []
        self.view2_scores_hat = []
        self.views_scores_subst = []
        self.views_scores_subst_int = []
        self.view_change_indexes = []
        self.view_change_index = 0

        self._number_of_figures = 1

    def run(self, parameter_values, update_figure_callback):
        self.stop_requested = False

        # UI parameters                
        path = parameter_values['input']
        first_tp = parameter_values['tp_start']
        last_tp = parameter_values['tp_end']
        reference_channel_fusion = parameter_values['reference_channel']
        views_to_fuse = [parameter_values.get('view_1_token', 'View1'), parameter_values.get('view_2_token', 'View2')]
        self.fused_token = parameter_values.get('fused_token', 'Fused')
        ome_glob_pattern = parameter_values.get('ome_glob_pattern', '*.ome')
        crop = parameter_values['crop']
        display_graphs = parameter_values['display_graphs']
        settings_to_fuse = parameter_values['settings_to_fuse']

        # verify that timepoint start is smaller than timepoint end
        if first_tp > last_tp:
            print('First timepoint cannot be larger than last timepoint.\nOperation stopped.')
            exit()

        # find all folders in directory and initialize the ones to copy and the ones to skip (e.g. the ones with max-projection)
        folders_to_scan = glob.glob(path + "/*")
        positions_in_folder = []
        folders_to_copy = []
        folders_to_skip = []
        settings_to_fuse_present = False
        for folder in folders_to_scan:
            folder_name = os.path.basename(folder)
            indices = [index for index in range(len(folder_name)) if folder_name.startswith('_', index)]
            if len(indices) == 0:
                folders_to_copy.append(folder)
            else:
                if len(indices) > 1 and folder_name.endswith('_max'):
                    folders_to_skip.append(folder)
                    position_name = folder_name[0:indices[-2]]
                else:
                    position_name = folder_name[0:indices[-1]]
                    settings_name = folder_name[indices[-1]+1:]
                    if settings_name == settings_to_fuse:
                        settings_to_fuse_present = True
                if not (position_name in positions_in_folder):
                    filenames_in_folder = glob.glob(folder + "/t0001_*.tif")
                    if len(filenames_in_folder) > 0:
                        positions_in_folder.append(position_name)
        if not settings_to_fuse_present:
            print('No settings to fuse. \nOperation stopped.')
            exit()
                      
        # list holding all the channels to fuse in each folder
        channels_in_folder_to_fuse = []
        # create directory to save fused data
        input_folder_name = os.path.basename(path)
        parent_path = os.path.abspath(os.path.join(path, os.pardir))
        requested_output = parameter_values.get('output_root', '<auto>')
        saving_path = os.path.join(parent_path, input_folder_name + '_fused') if requested_output in ('', '<auto>') else os.path.abspath(requested_output)
        creating_directory(saving_path)

        # create a file per each position, make user select the desired positions, and then remove all the files
        for position in positions_in_folder:
            with open(os.path.join(saving_path, position), 'w') as fp:
                pass
        supplied_positions = parameter_values.get('selected_positions', 'ALL')
        selected_positions_names = []
        if supplied_positions and supplied_positions.strip().upper() != 'ALL':
            selected_positions_names = [name.strip() for name in supplied_positions.split(',') if name.strip()]
            unknown = sorted(set(selected_positions_names) - set(positions_in_folder))
            for position in positions_in_folder:
                os.remove(os.path.join(saving_path, position))
            if unknown:
                os.rmdir(saving_path)
                print('Unknown position(s): ' + ', '.join(unknown) + '\nOperation stopped.')
                raise SystemExit(1)
        else:
            selected_positions = FileDialog.askopenfilenames(initialdir=saving_path, title="Select positions to process")
            if selected_positions == '' or selected_positions is None:
                os.rmdir(saving_path)
                print('No position to fuse. \nOperation stopped.')
                raise SystemExit(1)
            for position in positions_in_folder:
                os.remove(os.path.join(saving_path, position))
            selected_positions_names = [os.path.basename(position) for position in selected_positions]

        # select the folders to process
        folders_to_process = []
        folders_to_scan = [folder for folder in folders_to_scan if (folder not in folders_to_skip) and (folder not in folders_to_copy)]
        for position in positions_in_folder:
            if position in selected_positions_names:  
                for folder in folders_to_scan:
                    folder_name = os.path.basename(folder)
                    indices = [index for index in range(len(folder_name)) if folder_name.startswith('_', index)]
                    position_name = folder_name[0:indices[-1]]
                    settings_name = folder_name[indices[-1]+1:]
                    # if the folder is the one for the position
                    if position_name == position:
                        # if the settings is not to fuse, add to folders to process
                        if settings_name not in settings_to_fuse:
                            folders_to_process.append(folder)
                        else:
                            # check whether the reference channel for fusion is present in the folder
                            filenames_in_folder = glob.glob(folder + "/t0001_*.tif")
                            if len(filenames_in_folder) > 1 and (views_to_fuse[0] in filenames_in_folder[0]) and (views_to_fuse[1] in filenames_in_folder[1]):
                                reference_channel_fusion_present = False
                                for name in filenames_in_folder:
                                    name_start = name.find('t0001_') + 6
                                    name_end = name.find('_' + views_to_fuse[0] + '.tif')
                                    if (reference_channel_fusion == name[name_start:name_end]):
                                        reference_channel_fusion_present = True
                                        break
                                if reference_channel_fusion_present:
                                    folders_to_process.append(folder)
                                else:
                                    os.rmdir(saving_path)
                                    print('No reference channel found in ' + os.path.basename(folder) + '\nOperation stopped.')
                                    exit()
                            else:
                                folders_to_process.append(folder)
                
        # if no position is selected for fusion, delete "fusion" folder and stop script
        if len(folders_to_process) == 0:
            os.rmdir(saving_path)
            print('No position to fuse. \nOperation stopped.')
            exit()

        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time + ' Fusion started...')
        # copy the positions-to-copy in the destination directory
        for folder in folders_to_copy:
            imagenames_in_folder = glob.glob(folder + "/t*.tif")
            if len(imagenames_in_folder) == 0: # copy the folder if there are no images (e.g. settings folder)
                folder_name = os.path.basename(folder)
                destination_path = os.path.join(saving_path, folder_name)
                shutil.copytree(folder, destination_path)

        total_number_of_positions = len(folders_to_process)
        for folder in folders_to_process:
            # retrieve settings of folder
            indices = [index for index in range(len(folder)) if folder.startswith('_', index)]
            settings_name = folder[indices[-1]+1:]
            # clean indexes from previous run
            self.view_change_indexes.clear()
            # clear channel in folder to fuse from previous folder
            channels_in_folder_to_fuse.clear()
            # find ome companion in position directory
            file = glob.glob(os.path.join(folder, ome_glob_pattern))
            if len(file) == 0: 
                print('No OME file found in' + folder + '\nOperation stopped.')
                exit()
            elif len(file) > 1:
                print('Muliple OME files found in' + folder + '\nOperation stopped.')
                exit()
            else:

                with open(file[0], 'r') as file_read:
                    lines = file_read.readlines()
                for line in lines:
                    # convert crop in pixels
                    physical_size_x_start = line.find('PhysicalSizeX="')
                    if physical_size_x_start != -1:
                        physical_size_x_end = line.find('"', physical_size_x_start + 15)        
                        physical_size_x = float(line[physical_size_x_start + 15 : physical_size_x_end])
                        crop_pixels = int(crop/physical_size_x)
                    
                    size_x_start = line.find('SizeX="')
                    if size_x_start != -1:
                        size_x_end = line.find('"', size_x_start + 7)        
                        sizeX = int(line[size_x_start + 7 : size_x_end])
                        if sizeX < crop_pixels:
                            crop_pixels = sizeX
                        if parameter_values['save_as_crop']: crop_to_save = crop_pixels
                        else: crop_to_save = 0

                    size_y_start = line.find('SizeY="')
                    if size_y_start != -1:
                        size_y_end = line.find('"', size_y_start + 7)        
                        sizeY = int(line[size_y_start + 7 : size_y_end])
                        if sizeY < crop_pixels:
                            crop_pixels = sizeY
                        if parameter_values['save_as_crop']: crop_to_save = crop_pixels
                        else: crop_to_save = 0
                    
                    # found all channels to fuse in folder
                    if line.find('<Channel') != -1:
                        channel_name_start = line.find('Name="')
                        channel_name_end = line.find('" Color="', channel_name_start + 6)
                        channel_name = line[channel_name_start + 6 : channel_name_end]
                        if (channel_name.endswith('_' + views_to_fuse[0]) or channel_name.endswith('_' + views_to_fuse[1])) and (settings_name in settings_to_fuse):
                            channel_name = channel_name.rsplit('_', 1)[0]
                        channels_in_folder_to_fuse.append(channel_name)
                    # remove duplicate
                    channels_in_folder_to_fuse = list(set(channels_in_folder_to_fuse))

                # create list of timepoints to fuse
                imagenames_in_folder = glob.glob(folder + "/t*.tif")
                timepoints_in_folder_to_fuse = []
                for imagename in imagenames_in_folder:
                    name = os.path.basename(imagename)
                    timepoints_in_folder_to_fuse.append(int(name[1:5]))
                timepoints_in_folder_to_fuse = numpy.asarray(list(set(timepoints_in_folder_to_fuse)))
                if first_tp > timepoints_in_folder_to_fuse[-1]:
                    print('No timepoint ' + str(first_tp) + ' found in position \'' + os.path.basename(folder) + '\': position skipped')
                    continue
                timepoints_in_folder_to_fuse = timepoints_in_folder_to_fuse[timepoints_in_folder_to_fuse >= first_tp]
                if last_tp < timepoints_in_folder_to_fuse[-1]:
                    timepoints_in_folder_to_fuse = timepoints_in_folder_to_fuse[timepoints_in_folder_to_fuse <= last_tp]
                timepoints_in_folder_to_fuse = list(timepoints_in_folder_to_fuse)

                # create folder for fused data
                folder_name = os.path.basename(folder)
                output_path = os.path.join(saving_path, folder_name + '_fused')
                creating_directory(output_path)

                # generate random UUID for OME file and for the tiff files
                ome_uuid = str(uuid.uuid4())
                tif_uuid = []
                tif_uuid_needed = len(timepoints_in_folder_to_fuse) * len(channels_in_folder_to_fuse)
                if settings_name in settings_to_fuse:
                    tif_uuid_needed // 2
                for _ in range(0,tif_uuid_needed,1):
                    tif_uuid.append(str(uuid.uuid4()))
                
                # new channel index
                channel_indx = 1 
                # uuid index to attribute to each tif
                tif_uuid_indx = 0
                
                # make a copy of the ome in fused folder
                fileout = output_path + '//ome-tiff.companion.ome'
                copyfile(file[0], fileout)
                                            
                # copy of the .ome needs to be adapted to fused data
                with open(fileout, 'r') as file_read:
                    lines = file_read.readlines()

                with open(fileout, 'w') as file:
                    for line in lines:
                        write_line = False
                        # lines that are just copied
                        if line.find('<?') != -1 or line.find('</Pixels>') != -1 or line.find('</Image>') != -1 or line.find('</OME>') != -1:
                            write_line = True
                        # ID lines are updated and should always be rewriten
                        elif line.find('ID') != -1:
                            if parameter_values['save_as_crop']:
                                line = update_parameter(line=line, to_find='SizeX="', length=7, new_value=crop_pixels)
                                line = update_parameter(line=line, to_find='SizeY="', length=7, new_value=crop_pixels)
                            line = update_parameter(line=line, to_find='SizeC="', length=7, new_value=len(channels_in_folder_to_fuse)) # updata number of channel(s)
                            line = update_parameter(line=line, to_find='SizeT="', length=7, new_value=len(timepoints_in_folder_to_fuse)) # update number of timepoint(s)
                            if settings_name in settings_to_fuse:
                                line = update_parameter(line=line, to_find=views_to_fuse[0] + '"', length=len(views_to_fuse[0]), new_value=self.fused_token) # link to fused files
                            line = update_parameter(line=line, to_find='"urn:uuid:', length=10, new_value='"urn:uuid:' + ome_uuid) # link to fused files
                            write_line = True
                        
                        # update <Channel...> part of the companion file
                        if line.find('<Channel') != -1:
                            write_line = False
                            # get name in Channel part
                            name_start = line.find('Name="')
                            if name_start != -1:
                                name_end = line.find('"',name_start+6)
                                name = line[name_start:name_end+1]
                                # only write channel to fuse
                                for channel in channels_in_folder_to_fuse:
                                    found = name.find('%s' % (channel))
                                    if found != -1 and (not name.endswith('_' + views_to_fuse[1] + '"') or settings_name not in settings_to_fuse): 
                                        # get id of the channel
                                        id_c_start = line.find('"Channel:')
                                        if id_c_start != -1 and channel_indx < len(channels_in_folder_to_fuse)+1:
                                            id_c_end = line.find('"', id_c_start + 9)
                                            old_line = line[id_c_start:id_c_end+1]
                                            new_line = '"Channel:' + str(channel_indx) + '"'
                                            line = line.replace(old_line, new_line)
                                            channel_indx += 1
                                        
                                        write_line = True

                        # update <TiffData...> part of the companion file
                        if line.find('<TiffData') != -1:
                            if settings_name in settings_to_fuse:
                                line = update_parameter(line=line, to_find=views_to_fuse[0], length=len(views_to_fuse[0]), new_value=self.fused_token + '.tif')
                            write_line = False
                            # get file name in TiffData part
                            filename_start = line.find('FileName="')
                            if filename_start != -1:
                                filename_end = line.find('"',filename_start+10)
                                filename = line[filename_start:filename_end+1]
                                # if no channel to fuse or no timepoint to fuse is found, the line is not written
                                for timepoint in timepoints_in_folder_to_fuse:
                                    for channel in channels_in_folder_to_fuse:
                                        found = filename.find('t%0.4d_%s' % (timepoint, channel))
                                        if found != -1 and (not(filename.endswith('_' + views_to_fuse[1] + '.tif"')) or settings_name not in settings_to_fuse): 
                                            # update TIF uuid number of the ome file
                                            uuid_start = line.find('>urn:uuid:')
                                            correct_line = line.find('t%0.4d_%s' % (timepoint, channel))
                                            if uuid_start != -1 and correct_line!= -1:
                                                uuid_end = line.find('<', uuid_start + 10)
                                                old_line = line[uuid_start:uuid_end+1]
                                                new_line = '>urn:uuid:' + tif_uuid[tif_uuid_indx] + '<'
                                                line = line.replace(old_line, new_line)
                                                # take the next uuid
                                                tif_uuid_indx += 1

                                            if settings_name in settings_to_fuse:
                                                # update the id number of the channels in tif data
                                                tif_id_c_start = line.find('FirstC="')
                                                if tif_id_c_start != -1:
                                                    tif_id_c_end = line.find('"', tif_id_c_start + 8)
                                                    old_line = line[tif_id_c_start:tif_id_c_end+1]
                                                    curent_index = line[tif_id_c_start + 8 : tif_id_c_start + 9]
                                                    new_index = int(curent_index)//2
                                                    new_line = 'FirstC="'+ str(new_index) +'"'
                                                    line = line.replace(old_line, new_line)

                                            # update the id number of the timepoint in tif data (needs to always start at 0)
                                            tif_id_t_start = line.find('FirstT="')
                                            if tif_id_t_start != -1:
                                                tif_id_t_end = line.find('"', tif_id_t_start + 8)
                                                old_line = line[tif_id_t_start:tif_id_t_end+1]
                                                old_index = line[tif_id_t_start+8:tif_id_t_end]
                                                new_index = int(old_index) - first_tp + 1
                                                new_line = 'FirstT="'+ str(new_index) +'"'
                                                line = line.replace(old_line, new_line)
                                            
                                            write_line = True 
                        
                        # only write line if it passes criteria
                        if write_line: file.write(line)

                # Generate mask
                pixel, bin_factor = get_binned_pixel(physical_size_x)
                mask = dct_generate_mask(crop_pixels, crop_pixels, pixel)
                mask = _rebin(mask, bin_factor)
                
                # uuid index back to 0 to write tiff files
                tif_uuid_indx = 0

                # create folder for max projection data
                max_projection_output_path = os.path.join(saving_path, folder_name + '_fused_max')
                creating_directory(max_projection_output_path)
                
                # if there are not 2 views or the setting is not to fuse, only chosen timepoints are copied
                imagenames_in_folder = glob.glob(folder + "/t*_" + views_to_fuse[1] + ".tif")
                if len(imagenames_in_folder) == 0 or (settings_name not in settings_to_fuse):
                    for timepoint in timepoints_in_folder_to_fuse:
                        if self.stop_requested: 
                            print('\nStop requested by user.')
                            exit()
                        
                        for channel in channels_in_folder_to_fuse:
                            # open files to fuse
                            file_name = '%s/t%0.4d_%s.tif' % (folder, timepoint, channel)
                        
                            # verify that path to the file exist will not stop if timepoint is missing channels
                            if os.path.exists(file_name) :
                                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                print(time + ' Copying "' + str(os.path.basename(folder)) + '" (' + str(folders_to_process.index(folder) + 1) + '/' + str(total_number_of_positions) + '), channel "' + channel + '" (' + str(channels_in_folder_to_fuse.index(channel) + 1) + '/' + str(len(channels_in_folder_to_fuse)) + '), timepoint ' + str(timepoint) + ' (' + str(timepoints_in_folder_to_fuse.index(timepoint) + 1) + '/' + str(len(timepoints_in_folder_to_fuse)) + ')')
                                self.write_fused_file(file_name, file_name, timepoint, channel, output_path, max_projection_output_path, 0, ome_uuid, tif_uuid[tif_uuid_indx], crop_to_save)
                        
                    continue

                # fusing data per timepoints for the reference channel
                for timepoint in timepoints_in_folder_to_fuse:
                    if self.stop_requested: 
                        print('\nStop requested by user.')
                        exit()
                        
                    # re-initialize from previous computations
                    self.view1_scores = []
                    self.view2_scores = []
                    self.view1_scores_hat = []
                    self.view2_scores_hat = []
                    self.views_scores_subst = []
                    self.views_scores_subst_int = []
                    self.view_change_index = 0

                    # open files to fuse
                    file_name_view1 = '%s/t%0.4d_%s_%s.tif' % (folder, timepoint, reference_channel_fusion, views_to_fuse[0])
                    file_name_view2 = '%s/t%0.4d_%s_%s.tif' % (folder, timepoint, reference_channel_fusion, views_to_fuse[1])
                    
                    # verify that path to the file exist will not stop if timepoint is missing channels
                    if os.path.exists(file_name_view1) and os.path.exists(file_name_view2):
                        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(time + ' Fusing "' + str(os.path.basename(folder)) + '" (' + str(folders_to_process.index(folder) + 1) + '/' + str(total_number_of_positions) + '), channel "' + reference_channel_fusion + '" (1/' + str(len(channels_in_folder_to_fuse)) + '), timepoint ' + str(timepoint) + ' (' + str(timepoints_in_folder_to_fuse.index(timepoint) + 1) + '/' + str(len(timepoints_in_folder_to_fuse)) + ')')
                        
                        # find plane for change of view and get data to plot                        
                        result_find_channel = self.find_changing_plane(file_name_view1, file_name_view2, crop_pixels, mask, pixel, bin_factor)
                        self.view1_scores, self.view2_scores, self.view1_scores_hat, self.view2_scores_hat, self.views_scores_subst, self.views_scores_subst_int, self.view_change_index = result_find_channel
                        # show plot to determine change of view
                        if display_graphs:
                            update_figure_callback()
                        # save change of plane for other channels
                        self.view_change_indexes.append(self.view_change_index)  
                        # write the fused stack
                        self.write_fused_file(file_name_view1, file_name_view2, timepoint, reference_channel_fusion, output_path, max_projection_output_path, self.view_change_index, ome_uuid, tif_uuid[tif_uuid_indx], crop_to_save)
                        # write switching plane on report
                        annotate_report(crop, os.path.join(os.path.dirname(file_name_view1), os.path.basename(file_name_view1)), saving_path, self.view_change_index)

                # removes channel to fuse from channels in folder to fuse
                if reference_channel_fusion in channels_in_folder_to_fuse:
                    channels_in_folder_to_fuse.remove(reference_channel_fusion)

                # apply same change planes on others channels of the folder
                for timepoint in timepoints_in_folder_to_fuse:
                    if self.stop_requested:
                        print('\nStop requested by user.')
                        exit()

                    for channel in channels_in_folder_to_fuse:
                        # open files to fuse
                        file_name_view1 = '%s/t%0.4d_%s_%s.tif' % (folder, timepoint, channel, views_to_fuse[0])
                        file_name_view2 = '%s/t%0.4d_%s_%s.tif' % (folder, timepoint, channel, views_to_fuse[1])
                        
                        # verify that path to the file exist will not stop if timepoint is missing channels
                        if os.path.exists(file_name_view1) and os.path.exists(file_name_view2):
                            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            print(time + ' Fusing "' + str(os.path.basename(folder)) + '" (' + str(folders_to_process.index(folder) + 1) + '/' + str(total_number_of_positions) + '), channel "' + channel + '" (' + str(channels_in_folder_to_fuse.index(channel) + 1) + '/' + str(1 + len(channels_in_folder_to_fuse)) + '), timepoint ' + str(timepoint) + ' (' + str(timepoints_in_folder_to_fuse.index(timepoint) + 1) + '/' + str(len(timepoints_in_folder_to_fuse)) + ')')
                            self.write_fused_file(file_name_view1, file_name_view2, timepoint, channel, output_path, max_projection_output_path, self.view_change_indexes[timepoints_in_folder_to_fuse.index(timepoint)], ome_uuid, tif_uuid[tif_uuid_indx], crop_to_save)
        
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time + ' Fusion completed.')
        self.last_output_folder = saving_path
        return saving_path
    

    def stop(self):
        """
        Called by user interface to stop script.
        """
        self.stop_requested = True


    def get_figure_count(self, parameter_values):
        """
        Called by user interface to get the number of figures to generate.
        """
        return self._number_of_figures


    def update_figures(self, figures):
        """
        Update the figure(s).

        Parameters
        ----------
        figures : list
            List of the figure to draw on.
        """

        # plot scores and smoothed scores
        figures[0].clf()
        a1 = figures[0].add_subplot(1,1,1) 
        a1.grid()

        a1.plot(range(0, len(self.view1_scores), 1), self.view1_scores, '--', color='green')
        a1.plot(range(0, len(self.view2_scores), 1), self.view2_scores, '--', color='orange')
        a1.plot(range(0, len(self.view1_scores), 1), self.view1_scores_hat, color='green')
        a1.plot(range(0, len(self.view2_scores), 1), self.view2_scores_hat, color='orange')
        a1.plot(range(0, len(self.views_scores_subst), 1), self.views_scores_subst, color='yellow')
        a1.axvline(self.view_change_index - 1, linewidth=1, color='blue', label='Center')
        
        a2 = a1.twinx()
        a2.plot(range(0, len(self.views_scores_subst), 1), self.views_scores_subst_int, color='red')
        a2.axhline(criterion_threshold, linewidth=1, color='green', label='Center')
        
        figures[0].tight_layout()


    def find_changing_plane(self, file_name_view1, file_name_view2, crop, mask, pixel, bin_factor):
        with tifffile.TiffFile(file_name_view1) as stack1:
            depth = len(stack1.pages)
            image = stack1.pages[0]
            height, width = image.shape
            
            startx = width//2-(crop//2)
            starty = height//2-(crop//2)

            # score images and write the selected on in the tif file
            view1_scores = []
            view2_scores = []
            
            for z in range(0, depth, 1):
                if self.stop_requested: 
                    print('\nStop requested by user.')
                    exit()

                imarray_view1_crop = tifffile.imread(file_name_view1, key=z)
                imarray_view1_crop = imarray_view1_crop[starty:starty+crop,startx:startx+crop]
                
                imarray_view2_crop = tifffile.imread(file_name_view2, key=z)
                imarray_view2_crop = imarray_view2_crop[starty:starty+crop,startx:startx+crop]

                DCT_score_view1 = dct_fast(imarray_view1_crop, mask, pixel, bin_factor)
                DCT_score_view2 = dct_fast(imarray_view2_crop, mask, pixel, bin_factor)        
                
                view1_scores.append(DCT_score_view1)
                view2_scores.append(DCT_score_view2)

            # score curve smoothed median filtering
            if len(view1_scores) > 7:
                view1_scores_hat = medfilt(view1_scores, kernel_size=7)
                view2_scores_hat = medfilt(view2_scores, kernel_size=7)
            else:
                view1_scores_hat = numpy.copy(view1_scores)
                view2_scores_hat = numpy.copy(view2_scores)

            # substracting both views scores
            views_scores_subst = numpy.subtract(view1_scores_hat, view2_scores_hat)
            
            # integral starting from left
            views_scores_subst_int = []
            integral = 0
            for score in views_scores_subst:
                integral += score
                views_scores_subst_int.append(integral)

        # the center of the interval for which the integral is > threshold is the last plane of View1 
        max_contrast = numpy.max(views_scores_subst_int)
        min_contrast = numpy.min(views_scores_subst_int)
        views_scores_subst_int = (views_scores_subst_int - min_contrast) / (max_contrast - min_contrast)
        min_index = 0
        max_index = 0
        for z in range(0, depth):
            if views_scores_subst_int[z] > criterion_threshold:
                max_index = z 
                if min_index == 0:
                    min_index = z 
        view_change_index = (max_index + min_index) // 2
        # if view two is better move switching plane by one towards view 1
        if views_scores_subst[view_change_index] < 0:
            view_change_index -= 1
        view_change_index += 1 # plane index starts at 1

        return view1_scores, view2_scores, view1_scores_hat, view2_scores_hat, views_scores_subst, views_scores_subst_int, view_change_index
    

    def write_fused_file(self, file_name_view1, file_name_view2, timepoint, channel, output_path, max_projection_output_path, view_change_index, ome_uuid, tif_uuid, crop_to_save, output_filename=None, max_output_filename=None):
        fused_token = getattr(self, 'fused_token', 'Fused')
        if output_filename is not None:
            file_name_fused = os.path.join(output_path, output_filename)
            file_name_fused_max_proj = os.path.join(max_projection_output_path, max_output_filename or (Path(output_filename).stem + '_max.tif'))
        elif file_name_view1 != file_name_view2:
            file_name_fused = '%s/t%0.4d_%s_%s.tif' % (output_path, timepoint, channel, fused_token)
            file_name_fused_max_proj = '%s/t%0.4d_%s_%s_max.tif' % (max_projection_output_path, timepoint, channel, fused_token)
        else:
            file_name_fused = '%s/t%0.4d_%s.tif' % (output_path, timepoint, channel)
            file_name_fused_max_proj = '%s/t%0.4d_%s.tif' % (max_projection_output_path, timepoint, channel)

        with tifffile.TiffFile(file_name_view1) as tif_read_view1:
            with tifffile.TiffFile(file_name_view2) as tif_read_view2:
                with tifffile.TiffWriter(file_name_fused, bigtiff=True) as tif_write:
                    with tifffile.TiffWriter(file_name_fused_max_proj, bigtiff=True) as tif_write_max_proj:
                        # read shape from any of the stack open to create an empty array for the max projection
                        if crop_to_save == 0:
                            max_projection = numpy.zeros(tif_read_view2.pages[0].shape, dtype='uint16')
                        else:
                            max_projection = numpy.zeros((crop_to_save,crop_to_save), dtype='uint16')
                        
                        first_page=True
                        index_page = 0
                        for page_view1, page_view2 in zip(tif_read_view1.pages, tif_read_view2.pages):
                            if self.stop_requested: 
                                print('\nStop requested by user.')
                                exit()

                            if index_page<view_change_index:
                                image = page_view1.asarray()
                            else:
                                image = page_view2.asarray()
                            size_y, size_x = numpy.shape(image)
                            if crop_to_save != 0:
                                image = image[(size_y - crop_to_save)//2 : (size_y + crop_to_save)//2, (size_x - crop_to_save)//2 : (size_x + crop_to_save)//2]

                            if first_page and (270 in page_view2.tags.keys()):
                                tif_write.write(image, dtype='uint16', description = '<?xml version="1.0" encoding="UTF-8"?>\n'
                                                                    '<OME UUID="urn:uuid:' + ome_uuid + '" xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
                                                                    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                                                                    'xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 '
                                                                    'http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">\n'
                                                                    '<BinaryOnly MetadataFile="ome-tiff.companion.ome" UUID="urn:uuid:' + tif_uuid + '"/>\n'
                                                                    '</OME>')
                                first_page = False
                            else:
                                tif_write.write(image, dtype='uint16')
                            
                            max_projection = numpy.fmax(max_projection, image)
                            index_page += 1
                        
                        tif_write_max_proj.write(max_projection, dtype='uint16')



def _read_ome_dimensions(ome_path, crop_um, first_tiff):
    """Read only metadata/header fields; never load a complete volume."""
    attrs = {}
    if ome_path:
        with open(ome_path, 'r', encoding='utf-8', errors='replace') as fh:
            text = fh.read()
        for name in ('PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeZ',
                     'PhysicalSizeXUnit', 'PhysicalSizeYUnit', 'PhysicalSizeZUnit',
                     'SizeX', 'SizeY', 'SizeZ'):
            m = re.search(name + r'="([^"]+)"', text)
            if m:
                attrs[name] = m.group(1)
    with tifffile.TiffFile(str(first_tiff)) as tf:
        if len(tf.pages) > 1 and len(tf.pages[0].shape) == 2:
            page_y, page_x = tf.pages[0].shape
            page_z = len(tf.pages)
        else:
            series = tf.series[0]
            shape = tuple(int(v) for v in series.shape)
            if len(shape) != 3:
                raise ValueError(f"Expected a 3D TIFF for fusion, got {shape}: {first_tiff}")
            page_z, page_y, page_x = shape
    size_x = min(int(attrs.get('SizeX', page_x)), page_x)
    size_y = min(int(attrs.get('SizeY', page_y)), page_y)
    physical_size_x = float(attrs['PhysicalSizeX']) if 'PhysicalSizeX' in attrs else None
    if physical_size_x is None:
        raise ValueError(f"Cannot determine PhysicalSizeX from OME companion: {ome_path}")
    crop_pixels = min(int(float(crop_um) / physical_size_x), size_x, size_y)
    attrs.update({'SizeX': str(size_x), 'SizeY': str(size_y), 'SizeZ': str(page_z)})
    return attrs, max(1, crop_pixels)


def _write_pipeline_ome(path, position, records, source_attrs, output_xy, ome_uuid, file_uuids):
    """Write a compact fused companion with correct output filenames and indices."""
    ns = 'http://www.openmicroscopy.org/Schemas/OME/2016-06'
    ET.register_namespace('', ns)
    ome = ET.Element(f'{{{ns}}}OME', {'UUID': 'urn:uuid:' + ome_uuid})
    image = ET.SubElement(ome, f'{{{ns}}}Image', {'ID': 'Image:0', 'Name': str(position)})
    channels = sorted({str(r['channel']) for r in records})
    timepoints = sorted({str(r['timepoint']) for r in records}, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
    pixels_attrs = {
        'ID': 'Pixels:0', 'DimensionOrder': 'XYZCT', 'Type': 'uint16',
        'SizeX': str(output_xy[1]), 'SizeY': str(output_xy[0]),
        'SizeZ': str(source_attrs['SizeZ']), 'SizeC': str(len(channels)), 'SizeT': str(len(timepoints)),
    }
    for key in ('PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeZ',
                'PhysicalSizeXUnit', 'PhysicalSizeYUnit', 'PhysicalSizeZUnit'):
        if key in source_attrs:
            pixels_attrs[key] = str(source_attrs[key])
    pixels = ET.SubElement(image, f'{{{ns}}}Pixels', pixels_attrs)
    for ci, channel in enumerate(channels):
        ET.SubElement(pixels, f'{{{ns}}}Channel', {'ID': f'Channel:0:{ci}', 'Name': channel, 'SamplesPerPixel': '1'})
    record_by_key = {(str(r['timepoint']), str(r['channel'])): r for r in records}
    for ti, timepoint in enumerate(timepoints):
        for ci, channel in enumerate(channels):
            rec = record_by_key[(timepoint, channel)]
            tiff_data = ET.SubElement(pixels, f'{{{ns}}}TiffData', {
                'FirstZ': '0', 'FirstC': str(ci), 'FirstT': str(ti), 'PlaneCount': str(source_attrs['SizeZ'])
            })
            uuid_elem = ET.SubElement(tiff_data, f'{{{ns}}}UUID', {'FileName': rec['output_name']})
            uuid_elem.text = 'urn:uuid:' + file_uuids[rec['output_name']]
    ET.ElementTree(ome).write(path, encoding='utf-8', xml_declaration=True)


def run_pipeline_jobs(jobs_path, output_root, reference_channel, crop_um=950, save_as_crop=False,
                      fused_token='Fused', overwrite=False, max_projection_root=None):
    """Headless wrapper for exact file pairs; fusion/switching-plane math is unchanged."""
    jobs = json.loads(Path(jobs_path).read_text(encoding='utf-8'))
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    max_root = Path(max_projection_root).resolve() if max_projection_root else output_root.parent / 'logs' / 'fusion_max'
    script = Script()
    script.stop_requested = False
    script.fused_token = fused_token
    result = {'output_root': str(output_root), 'positions': {}, 'status': 'success'}

    for position, pdata in jobs.get('positions', {}).items():
        records = pdata.get('records', [])
        if not records:
            raise ValueError(f"[{position}] no fusion records")
        out_dir = output_root / position
        max_dir = max_root / position
        out_dir.mkdir(parents=True, exist_ok=True)
        max_dir.mkdir(parents=True, exist_ok=True)
        existing_outputs = [out_dir / r['output_name'] for r in records if (out_dir / r['output_name']).exists()]
        if existing_outputs and not overwrite:
            raise FileExistsError(
                f"[{position}] {len(existing_outputs)} fusion output(s) already exist. "
                "Use pipeline resume for a validated skip or enable overwrite; partial silent reuse is not allowed."
            )
        ome_path = pdata.get('ome_path')
        if ome_path:
            shutil.copy2(ome_path, out_dir / 'source_ome_companion.ome')
        source_attrs, crop_pixels = _read_ome_dimensions(ome_path, crop_um, records[0]['view1'])
        physical_size_x = float(source_attrs['PhysicalSizeX'])
        crop_to_save = crop_pixels if save_as_crop else 0
        pixel, bin_factor = get_binned_pixel(physical_size_x)
        mask = _rebin(dct_generate_mask(crop_pixels, crop_pixels, pixel), bin_factor)

        by_tp = {}
        channels = sorted({str(r['channel']) for r in records})
        selected_reference = str(reference_channel or '')
        if not selected_reference:
            if len(channels) != 1:
                raise ValueError(f"[{position}] reference channel is ambiguous; detected {channels}")
            selected_reference = channels[0]
        for rec in records:
            by_tp.setdefault(str(rec['timepoint']), []).append(rec)

        pos_results = []
        position_ome_uuid = str(uuid.uuid4())
        file_uuids = {r['output_name']: str(uuid.uuid4()) for r in records}
        for timepoint in sorted(by_tp, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))):
            tp_records = by_tp[timepoint]
            refs = [r for r in tp_records if str(r['channel']) == selected_reference]
            if len(refs) != 1:
                raise ValueError(f"[{position}] timepoint {timepoint}: expected one reference-channel pair for {selected_reference!r}, found {len(refs)}")
            ref = refs[0]
            print(f"[{position}] timepoint {timepoint}: estimating switching plane from channel {selected_reference}", flush=True)
            found = script.find_changing_plane(ref['view1'], ref['view2'], crop_pixels, mask, pixel, bin_factor)
            script.view1_scores, script.view2_scores, script.view1_scores_hat, script.view2_scores_hat, script.views_scores_subst, script.views_scores_subst_int, view_change_index = found
            for rec in sorted(tp_records, key=lambda r: str(r['channel'])):
                output_name = rec['output_name']
                output_path = out_dir / output_name
                print(f"[{position}] fusing t={timepoint}, channel={rec['channel']} -> {output_name}", flush=True)
                script.write_fused_file(
                    rec['view1'], rec['view2'], int(timepoint) if str(timepoint).isdigit() else 0,
                    str(rec['channel']), str(out_dir), str(max_dir), view_change_index,
                    position_ome_uuid, file_uuids[output_name], crop_to_save,
                    output_filename=output_name,
                )
                pos_results.append({'timepoint': timepoint, 'channel': rec['channel'], 'output': str(output_path), 'status': 'ok', 'switching_plane': view_change_index})
        output_side = crop_pixels if save_as_crop else int(source_attrs['SizeX'])
        output_y = crop_pixels if save_as_crop else int(source_attrs['SizeY'])
        _write_pipeline_ome(out_dir / 'ome-tiff.companion.ome', position, records, source_attrs, (output_y, output_side), position_ome_uuid, file_uuids)
        result['positions'][position] = {'ome_source': ome_path, 'reference_channel': selected_reference, 'results': pos_results}
    manifest_path = output_root / 'fusion_manifest.json'
    manifest_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"Actual fusion output folder: {output_root}", flush=True)
    print(f"Wrote fusion manifest: {manifest_path}", flush=True)
    return str(output_root)


def _headless_parser():
    p = argparse.ArgumentParser(description='Headless exact-pair wrapper around the existing fuse_views algorithm.')
    p.add_argument('--pipeline-jobs', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--reference-channel', default='')
    p.add_argument('--crop', type=int, default=950)
    p.add_argument('--save-as-crop', action='store_true')
    p.add_argument('--fused-token', default='Fused')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--max-projection-root', default=None)
    return p


if __name__ == '__main__':
    if '--pipeline-jobs' in sys.argv:
        args = _headless_parser().parse_args()
        run_pipeline_jobs(
            jobs_path=args.pipeline_jobs,
            output_root=args.output_root,
            reference_channel=args.reference_channel,
            crop_um=args.crop,
            save_as_crop=args.save_as_crop,
            fused_token=args.fused_token,
            overwrite=args.overwrite,
            max_projection_root=args.max_projection_root,
        )
    else:
        python_interpreter = sys.executable
        script_gui = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'gui', 'script_gui.py'))
        script_name = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]
        os.spawnl(os.P_WAIT, python_interpreter, python_interpreter, script_gui, script_name)
