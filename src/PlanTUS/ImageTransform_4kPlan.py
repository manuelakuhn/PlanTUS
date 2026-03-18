#!/usr/bin/env python3

import ants
from ants.utils import nibabel_nifti_to_ants
from bids_validator import BIDSValidator
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform
import numpy as np

import argparse
from pathlib import Path
import re

class ImgTransf:
    """
    Transform t1 image from subject space to
    MNI space and set origin as requested by k-Plan
    """
    def __init__(self, T1, outdir, mni=None, fname=None, bids=False):
        self.bids = bids
        if self.bids:
            self.is_valid(T1)
        self.T1 = T1
        self.outdir = Path(outdir)
        if mni is not None:
            self.mni = ants.image_read(mni)
        else:
            self.mni = mni
        self.fname = fname

    def is_valid(self, T1):
        """
        Check BIDS dataset compliancy.
        Given that bids_validator accepts only
        the Path relative to the BIDS root
        directory preceded by a /, this method
        reconstructs the bids path starting
        from the t1 file name to avoid
        annoying side effects for the user.
        It then checks existence of the path
        and if there it checks the bids validity

        Parameters
        ----------
        T1 : str, path of the t1 image
        """
        T1_name = Path(T1).name
        sub_label = re.findall(r"sub-(.*?)_", T1_name)
        bids_dir = Path(f'sub-{sub_label[0]}') / 'anat' / T1_name
        if bids_dir.absolute().exists():
            validator = BIDSValidator()
            if not validator.is_bids(f'/{str(bids_dir)}'):
                raise ValueError("T1 file name must be BIDS compliant")
        else:
            raise ValueError('The bids directory does not exists')

    def _t1_to_mni(self):
        """
        Register and normalize T1
        image to the chosen MNI image

        Returns
        -------

        T1_transformed_nib : nib.image T1 in MNI space
        """
        T1_ants = ants.image_read(self.T1)
        ants_reg = ants.registration(
                fixed=self.mni,
                moving=T1_ants,
                type_of_transform='Rigid'
                )
        T1_transformed = ants.apply_transforms(
                fixed=self.mni,
                moving=T1_ants,
                transformlist=ants_reg['fwdtransforms'],
                interpolator='linear'
                )
        T1_transformed_nib = nibabel_nifti_to_ants.to_nibabel_nifti(
                T1_transformed
                )
        return T1_transformed_nib

    def _set_origin(self, T1_acpc):
        """
        Set left, inferior,
        posterior corner
        of image to (0,0,0)

        Parameters
        ----------
        T1_acpc : nib.image T1 in MNI space

        Returns
        -------
        T1_acpc_kPlan : nib.image T1 with
                        k-Plan compatible
                        affine matrix
        """
        print(
        (f"Original affine matrix:\n"
         f"{T1_acpc.affine}")
        )
        # Get canonical affine matrix
        T1_canonical = nib.as_closest_canonical(T1_acpc)
        # Create new affine matrix from canonical affine matrix
        # + set origin to (0,0,0)
        affine_kPlan = T1_canonical.affine
        affine_kPlan[0:3,3] = np.array([0,0,0])
        # Create new T1 image with new affine matrix
        header_kPlan = T1_acpc.header.copy()
        data_kPlan = T1_acpc.get_fdata()
        T1_acpc_kPlan = nib.nifti1.Nifti1Image(
                data_kPlan,
                affine_kPlan,
                header=header_kPlan
                )
        print(
        ((f"New affine matrix:\n"
        f"{T1_acpc_kPlan.affine}"))
        )
        return T1_acpc_kPlan

    def _mod_orientation(self, img):
        """
        If the image is not oriented as RAS,
        it gets re-oriented. Most likely
        un-necessary given nib.as_closest_canonical()
        used in the previous method.

        Parameters
        ----------
        img : nib.image

        Returns
        -------
        img : nib.image with RAS orientation
        """
        if nib.aff2axcodes(img.affine) != ('R', 'A', 'S'):
            print('reorienting T1 ...')
            orig_or = nib.io_orientation(img.affine)
            targ_or = axcodes2ornt(('R', 'A', 'S'))

            transform = ornt_transform(orig_or, targ_or)
            canon_or_img = img.as_reoriented(transform)
            print(
            (f"to {nib.aff2axcodes(canon_or_img.affine)}\n"
            f"{canon_or_img.affine}")
            )
            return canon_or_img
        return img

    def _to_bids(self, img):
        """
        Make output file BIDS compatible
        """
        if self.bids:
            space = "individual"
            if self.mni is not None:
                space = "MNI"
            source_entities = re.findall(r'.*?(?=T1w)', Path(self.T1).name)
            T1_name = f"{source_entities}_space-{space}_desc-4kplan_T1w.nii.gz"
            filename = self.outdir / T1_name
        else:
            if self.fname is not None:
                filename = self.outdir / f'{self.fname}.nii.gz'
            else:
                filename = self.outdir / 'T1w_acpc_kplan.nii.gz'
        img.to_filename(filename)

    def __call__(self):
        """
        Make class callable
        and call all the necessary
        methods
        """
        # Normalize to MNI
        if self.mni is not None:
            print("-------------------------")
            print("Registration to MNI space")
            print("-------------------------")
            T1 = self._t1_to_mni()
        else:
            print("---------------------")
            print("Keeping subject space")
            print("---------------------")
            T1 = nib.load(self.T1)
            print(f"Original image dimensions: {T1.header.get_data_shape()}")
            print(f"Data type: {T1.header.get_data_dtype()}")
        # Modify affine matrix
        T1_kplan = self._set_origin(T1)
        # If orientation is not RAS change it
        T1_kplan = self._mod_orientation(T1_kplan)
        print(f"Transformed image dimensions: {T1_kplan.header.get_data_shape()}")
        print(f"Data type: {T1_kplan.header.get_data_dtype()}")
        self._to_bids(T1_kplan)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-t1",
            "--t1_image",
            type=str,
            required=True,
            help=(
                't1 image path, if the bids option is used, '
                'the t1 image must be in a valid bids dataset')
            )
    parser.add_argument(
            '-o',
            '--outdir',
            type=Path,
            required=True,
            help='output directory'
            )
    parser.add_argument(
            "-m",
            "--mni_image",
            type=str,
            required=False,
            help='mni image path (optional)'
            )
    parser.add_argument(
            "-f",
            "--fname",
            type=str,
            required=False,
            help='output file name (optional)'
            )

    parser.add_argument(
            '-b',
            '--bids',
            action="store_true",
            help=("if specified it requires a bids compliant dataset "
                  "and returns bids compliant files")
            )
    args = parser.parse_args()
    IT = ImgTransf(
            args.t1_image,
            args.outdir,
            args.mni_image,
            args.fname,
            args.bids
            )
    IT()

if __name__ == '__main__':
    main()
