# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

import os
import collections
import numpy
import scipy.sparse
from tvb.basic.traits import util
from tvb.basic.readers import try_get_absolute_path, FileReader
from tvb.basic.traits import core
from tvb.basic.logger.builder import get_logger
from . import local_connectivity, region_mapping, surfaces

LOG = get_logger(__name__)


class Cortex(surfaces.CorticalSurface):
    """
    Wrapper Class over a CorticalSurface, to be used when preparing a simulation launch.
    """

    _ui_complex_datatype = surfaces.CorticalSurface

    _ui_name = "A cortex..."

    region_areas = None
    region_orientation = None

    # A factor that rescales local connectivity strengths.
    coupling_strength = numpy.array([1.0], dtype=numpy.float64)

    # A 2-D array which projects the neural activity on the cortical
    # surface to a set of EEG sensors.
    eeg_projection = numpy.array([], dtype=numpy.float64)
    #  requires linked sensors.SensorsEEG


    # A 2-D array which projects the neural activity on the cortical
    # surface to a set of MEG sensors.
    meg_projection = numpy.array([], dtype=numpy.float64)
    #  requires linked SensorsMEG

    # A 2-D array which projects the neural activity on the
    # cortical surface to a set of embeded sensors.
    internal_projection = numpy.array([], dtype=numpy.float64)
    #  requires linked SensorsInternal

    def __init__(self, local_connectivity=None, region_mapping_data=None, *args, **kwargs):
        # "Define the interaction between neighboring network nodes. This is implicitly integrated in"
        #     " the definition of a given surface as an excitatory mean coupling of directly adjacent neighbors to"
        #     " the first state variable of each population model (since these typically represent the mean-neural"
        #     " membrane voltage). This coupling is instantaneous (no time delays)."
        self.local_connectivity = local_connectivity
        self.region_mapping_data = region_mapping_data
        super(Cortex, self).__init__(*args, **kwargs)


    def populate_cortex(self, cortex_surface, cortex_parameters=None):
        """
        Populate 'self' from a CorticalSurfaceData instance with additional
        CortexData specific attributes.

        :param cortex_surface:  CorticalSurfaceData instance
        :param cortex_parameters: dictionary key:value, where key is attribute on CortexData
        """
        for name in cortex_surface.trait:
            try:
                setattr(self, name, getattr(cortex_surface, name))
            except Exception as exc:
                self.logger.exception(exc)
                self.logger.error("Could not set attribute '" + name + "' on Cortex")
        for key, value in cortex_parameters.iteritems():
            setattr(self, key, value)
        return self

    @property
    def region_mapping(self):
        """
        Define shortcut for retrieving RegionMapping map array.
        """
        if self.region_mapping_data is None:
            return None
        return self.region_mapping_data.mapping


    def configure(self):
        """
        Invoke the compute methods for computable attributes that haven't been
        set during initialisation.
        """
        super(Cortex, self).configure()

        if self.region_orientation is None:
            self.compute_region_orientation()

        if self.region_areas is None:
            self.compute_region_areas()

        if self.local_connectivity is None:
            self.local_connectivity = local_connectivity.LocalConnectivity(cutoff=40.0, surface=self)

        if self.local_connectivity.matrix is None:
            self.compute_local_connectivity()

        # Pad the local connectivity matrix with zeros when non-cortical regions
        # are included in the long range connectivity...
        if self.local_connectivity.matrix.shape[0] < self.region_mapping.shape[0]:
            LOG.info("There are non-cortical regions, will pad local connectivity")
            padding = scipy.sparse.csc_matrix((self.local_connectivity.matrix.shape[0],
                                               self.region_mapping.shape[0] - self.local_connectivity.matrix.shape[0]))
            self.local_connectivity.matrix = scipy.sparse.hstack([self.local_connectivity.matrix, padding])

            padding = scipy.sparse.csc_matrix((self.region_mapping.shape[0] - self.local_connectivity.matrix.shape[0],
                                               self.local_connectivity.matrix.shape[1]))
            self.local_connectivity.matrix = scipy.sparse.vstack([self.local_connectivity.matrix, padding])

    def _find_summary_info(self):
        """
        Extend the base class's scientific summary information dictionary.
        """
        summary = super(Cortex, self)._find_summary_info()
        summary["Number of regions"] = numpy.sum(self.region_areas > 0.0)
        summary["Region area, mean (mm:math:`^2`)"] = self.region_areas.mean()
        summary["Region area, minimum (mm:math:`^2`)"] = self.region_areas.min()
        summary["Region area, maximum (mm:math:`^2`)"] = self.region_areas.max()

        return summary

    def compute_local_connectivity(self):
        """
        """
        LOG.info("Computing local connectivity matrix")
        loc_con_cutoff = self.local_connectivity.cutoff
        self.compute_geodesic_distance_matrix(max_dist=loc_con_cutoff)

        self.local_connectivity.matrix_gdist = self.geodesic_distance_matrix.copy()
        self.local_connectivity.compute()  # Evaluate equation based distance
        # self.local_connectivity.trait["matrix"].log_debug(owner=self.__class__.__name__ + ".local_connectivity")

        #HACK FOR DEBUGGING CAUSE TRAITS REPORTS self.local_connectivity.trait["matrix"] AS BEING EMPTY...
        lcmat = self.local_connectivity.matrix
        sts = str(lcmat.__class__)
        shape = str(lcmat.shape)
        sparse_format = str(lcmat.format)
        nnz = str(lcmat.nnz)
        dtype = str(lcmat.dtype)
        if lcmat.data.any() and lcmat.data.size > 0:
            array_max = lcmat.data.max()
            array_min = lcmat.data.min()
        else:
            array_max = array_min = 0.0

    def compute_region_areas(self):
        """Update the region_area attribute."""
        regions = numpy.unique(self.region_mapping)
        number_of_regions = len(regions)
        region_surface_area = numpy.zeros((number_of_regions, 1))
        avt = numpy.array(self.vertex_triangles)
        #NOTE: Slightly overestimates as it counts overlapping border triangles,
        #      but, not really a problem provided triangle-size << region-size.

        #NOTE: Check if there are non-cortical regions.

        if len(self.region_mapping) > len(self.vertex_normals):
            vertices_per_region = numpy.bincount(self.region_mapping)
            # Assume non-cortical regions will have len 1.
            non_cortical_regions, = numpy.where(vertices_per_region == 1)
            cortical_regions, = numpy.where(vertices_per_region > 1)
            #Average orientation of the region
            cortical_region_mapping = [x for x in self.region_mapping if x in cortical_regions]

            for nk in non_cortical_regions:
                region_surface_area[nk, :] = 0.0
            for k in cortical_regions:
                regs = map(set, avt[cortical_region_mapping == k])
                region_triangles = set.union(*regs)
                region_surface_area[k] = self.triangle_areas[list(region_triangles)].sum()
        else:
            for k in regions:
                regs = map(set, avt[self.region_mapping == k])
                region_triangles = set.union(*regs)
                region_surface_area[k] = self.triangle_areas[list(region_triangles)].sum()

        util.log_debug_array(LOG, region_surface_area, "region_areas", owner=self.__class__.__name__)
        self.region_areas = region_surface_area

    def compute_region_orientation(self):
        """Update the region_orientation attribute."""
        regions = numpy.unique(self.region_mapping)
        average_orientation = numpy.zeros((len(regions), 3))
        if len(self.region_mapping) > len(self.vertex_normals):
            # Count how many vertices each region has.
            counter = collections.Counter(self.region_mapping)
            # Presumably non-cortical regions will have len 1 vertex assigned.
            vertices_per_region = numpy.asarray(counter.values())
            non_cortical_regions = numpy.where(vertices_per_region == 1)
            cortical_regions = numpy.where(vertices_per_region > 1)
            cortical_region_mapping = [x for x in self.region_mapping if x in cortical_regions[0]]
            #Average orientation of the region
            for k in cortical_regions[0]:
                orient = self.vertex_normals[cortical_region_mapping == k, :]
                avg_orient = numpy.mean(orient, axis=0)
                average_orientation[k, :] = avg_orient / numpy.sqrt(numpy.sum(avg_orient ** 2))
            for nk in non_cortical_regions[0]:
                average_orientation[nk, :] = numpy.zeros((1, 3))
        else:
            #Average orientation of the region
            for k in regions:
                orient = self.vertex_normals[self.region_mapping == k, :]
                avg_orient = numpy.mean(orient, axis=0)
                average_orientation[k, :] = avg_orient / numpy.sqrt(numpy.sum(avg_orient ** 2))

        util.log_debug_array(LOG, average_orientation, "region_orientation", owner=self.__class__.__name__)
        self.region_orientation = average_orientation

    @classmethod
    def from_file(cls, source_file="cortex_16384.zip",
                  region_mapping_file=None,
                  local_connectivity_file=None, eeg_projection_file=None, instance=None):

        result = super(Cortex, cls).from_file(source_file, instance)

        return result

    @staticmethod
    def from_file_projection_array(source_file="projection_eeg_62_surface_16k.mat",
                                   matlab_data_name="ProjectionMatrix"):

        source_full_path = try_get_absolute_path("tvb_data.projectionMatrix", source_file)
        reader = FileReader(source_full_path)

        return reader.read_array(matlab_data_name=matlab_data_name)
