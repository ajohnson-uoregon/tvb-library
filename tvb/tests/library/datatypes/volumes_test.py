# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#
"""
Created on Mar 20, 2013

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import numpy
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.datatypes.volumes import Volume, StructuralMRI


class TestVolumes(BaseTestCase):
    """
    Tests the defaults for `tvb.datatypes.volumes` module.
    """

    def test_volume(self):
        dt = Volume()
        summary_info = dt._find_summary_info()
        assert summary_info['Origin'].shape == (0,)
        assert summary_info['Voxel size'].shape == (0,)
        assert summary_info['Volume type'] == 'Volume'
        assert summary_info['Units'] == 'mm'
        assert dt.origin.shape == (0,)
        assert dt.voxel_size.shape == (0,)
        assert dt.voxel_unit == 'mm'

    def test_get_min_max(self):
        dt = Volume(array_data=numpy.array(range(30)))
        assert dt.get_min_max_values() == (0, 29)

    def test_structural_MRI(self):
        dt = StructuralMRI()

        # check that inheritace happens
        summary_info = dt._find_summary_info()
        assert summary_info['Origin'].shape == (0,)
        assert summary_info['Voxel size'].shape == (0,)
        assert summary_info['Volume type'] == 'StructuralMRI'
        assert summary_info['Units'] == 'mm'
        assert dt.origin.shape == (0,)
        assert dt.voxel_size.shape == (0,)
        assert dt.voxel_unit == 'mm'

        assert dt.weighting == ""
