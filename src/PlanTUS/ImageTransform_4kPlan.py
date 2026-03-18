#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maximilian Lueckel, mlueckel@uni-mainz.de
"""
#=============================================================================
#=============================================================================
# Transform a T1 image for compatibility with k-Plan and Localite
#=============================================================================
#=============================================================================
# Input:
# T1 image in native/subject space
#-----------------------------------------------------------------------------
# Output:
# ACPC-alinged T1 image with left, posterior, inferior corner set as origin
# (0,0,0).
#=============================================================================
#=============================================================================
# Requirements:
# Python
# - numpy
# - nibabel
# - ants (pip install antspyx)
# FSL
#=============================================================================
# Specify variables
#=============================================================================

# Path to T1 image
T1_filepath = '/path/to/T1_image.nii.gz'

# Path to MNI template (should have same/similar resolution as your input T1 image)
MNI_template_filepath = '/path/to/MNI_template.nii.gz'


#=============================================================================
#=============================================================================
# Import necessary packages
#=============================================================================
#=============================================================================
import os
import numpy as np
import nibabel as nib
import ants
#=============================================================================
#=============================================================================
# Align input T1 image to ACPC
#=============================================================================
#=============================================================================

# Get path to T1 iamge
T1_path = os.path.split(T1_filepath)[0]
T1_filename = os.path.split(T1_filepath)[0]

# Load T1 image and MNI template
T1 = ants.image_read(T1_filepath)
MNI_template = ants.image_read(MNI_template_filepath)

# Register T1 image to MNI template using ANTs
ants_registration = ants.registration(fixed=MNI_template,
                                      moving=T1,
                                      type_of_transform='Rigid')

T1_transformed = ants.apply_transforms(fixed=MNI_template,
                                       moving=T1,
                                       transformlist=ants_registration['fwdtransforms'],
                                       interpolator='linear')

T1_transformed_nib = ants.to_nibabel(T1_transformed)

T1_transformed_nib.to_filename(T1_path + '/T1w_acpc.nii.gz')


#=============================================================================
#=============================================================================
# Set left, inferior, posterior corner of image to (0,0,0)
#=============================================================================
#=============================================================================

# Load ACPC-aligned T1 image
T1_acpc = nib.load(T1_path + '/T1w_acpc.nii.gz')

# Get current affine matrix
affine_orig = T1_acpc.affine

# Get canonical affine matrix
T1_canonical = nib.as_closest_canonical(T1_acpc)
affine_canonical = T1_canonical.affine

# Create new affine matrix from canonical affine matrix + set origin to (0,0,0)
affine_kPlan = T1_canonical.affine
affine_kPlan[0:3,3] = np.array([0,0,0])

# Create new T1 image with new affine matrix
header_kPlan = T1_acpc.header.copy()
data_kPlan = T1_acpc.get_fdata()
T1_acpc_kPlan = nib.nifti1.Nifti1Image(data_kPlan, affine_kPlan, header=header_kPlan)

# Save new T1 image
T1_acpc_kPlan.to_filename(T1_path + '/T1w_acpc_kPlan.nii.gz')

# If one of the diagonal elements of the affine matrix is negative,
# additionally swap the respective dimension(s) using FSL

# x-dimension
if affine_orig[0,0] < 0:
    os.system('fslswapdim' + ' ' +
              T1_path + '/T1w_acpc_kPlan.nii.gz' + ' ' +
              '-x y z' + ' ' +
              T1_path + '/T1w_acpc_kPlan.nii.gz')

# y-dimenstion
if affine_orig[1,1] < 0:
    os.system('fslswapdim' + ' ' +
              T1_path + '/T1w_acpc_kPlan.nii.gz' + ' ' +
              'x -y z' + ' ' +
              T1_path + '/T1w_acpc_kPlan.nii.gz')

# z-dimension
if affine_orig[2,2] < 0:
    os.system('fslswapdim' + ' ' +
              T1_path + '/T1w_acpc_kPlan.nii.gz' + ' ' +
              'x y -z' + ' ' +
              T1_path + '/T1w_acpc_kPlan.nii.gz')
