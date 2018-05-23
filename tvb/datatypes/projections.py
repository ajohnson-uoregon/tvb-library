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
The ProjectionMatrices DataTypes. This brings together the scientific and framework
methods that are associated with the surfaces data.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from tvb.basic.readers import try_get_absolute_path, FileReader
from tvb.datatypes import surfaces, sensors
from tvb.basic.traits.types_mapped import MappedType
import numpy


EEG_POLYMORPHIC_IDENTITY = "projEEG"
MEG_POLYMORPHIC_IDENTITY = "projMEG"
SEEG_POLYMORPHIC_IDENTITY = "projSEEG"


class ProjectionMatrix(object):
    """
    Base DataType for representing a ProjectionMatrix.
    The projection is between a source of type CorticalSurface and a set of Sensors.
    """

    projection_type = ""


    def __init__(self, brain_skull=None, skull_skin=None, skin_air=None,
                 conductances=None, sources=None, sensors=None,
                 projection_data=None, load_file=None, *args, **kwargs):
        self.brain_skull = brain_skull
        self.skull_skin = skull_skin
        self.skin_air = skin_air

        if conductances is None:
            conductances = {'air': 0.0, 'skin': 1.0, 'skull': 0.01, 'brain': 1.0}
        self.conductances = conductances

        self.sources = sources
        self.sensors = sensors

        if load_file is not None:
            projection_data = ProjectionMatrix.from_file(source_file=load_file)

        self.projection_data = projection_data


    @property
    def shape(self):
        return self.projection_data.shape


    @classmethod
    def from_file(cls, source_file=None, matlab_data_name=None, is_brainstorm=False, instance=None):

        source_full_path = try_get_absolute_path("tvb_data.projectionMatrix", source_file)
        reader = FileReader(source_full_path)
        if is_brainstorm:
            projection_data = reader.read_gain_from_brainstorm()
        else:
            projection_data = reader.read_array(matlab_data_name=matlab_data_name)
        return projection_data


class ProjectionSurfaceEEG(ProjectionMatrix):
    """
    Specific projection, from a CorticalSurface to EEG sensors.
    """

    __tablename__ = None

    __mapper_args__ = {'polymorphic_identity': EEG_POLYMORPHIC_IDENTITY}

    def __init__(self, load_file=None, *args, **kwargs):
        self.projection_type = EEG_POLYMORPHIC_IDENTITY
        self.sensors = sensors.SensorsEEG

        if load_file is not None:
            projection_data = ProjectionSurfaceEEG.from_file(source_file=load_file)
        else:
            projection_data = None

        super(ProjectionSurfaceEEG, self).__init__(*args, projection_data=projection_data, **kwargs)

    @classmethod
    def from_file(cls, source_file='projection_eeg_65_surface_16k.npy', matlab_data_name="ProjectionMatrix",
                  is_brainstorm=False, instance=None):

        source_full_path = try_get_absolute_path("tvb_data.projectionMatrix", source_file)
        reader = FileReader(source_full_path)
        if is_brainstorm:
            projection_data = reader.read_gain_from_brainstorm()
        else:
            projection_data = reader.read_array(matlab_data_name=matlab_data_name)
        return projection_data


class ProjectionSurfaceMEG(ProjectionMatrix):
    """
    Specific projection, from a CorticalSurface to MEG sensors.
    """

    __tablename__ = None

    __mapper_args__ = {'polymorphic_identity': MEG_POLYMORPHIC_IDENTITY}

    def __init__(self, load_file=None, *args, **kwargs):
        self.projection_type = MEG_POLYMORPHIC_IDENTITY
        self.sensors = sensors.SensorsMEG

        if load_file is not None:
            projection_data = ProjectionSurfaceMEG.from_file(source_file=load_file)
        else:
            projection_data = None

        super(ProjectionSurfaceMEG, self).__init__(*args, projection_data=projection_data, **kwargs)

    @classmethod
    def from_file(cls, source_file='projection_meg_276_surface_16k.npy', matlab_data_name=None, is_brainstorm=False,
                  instance=None):
        source_full_path = try_get_absolute_path("tvb_data.projectionMatrix", source_file)
        reader = FileReader(source_full_path)
        if is_brainstorm:
            projection_data = reader.read_gain_from_brainstorm()
        else:
            projection_data = reader.read_array(matlab_data_name=matlab_data_name)
        return projection_data


class ProjectionSurfaceSEEG(ProjectionMatrix):
    """
    Specific projection, from a CorticalSurface to SEEG sensors.
    """

    __tablename__ = None

    __mapper_args__ = {'polymorphic_identity': SEEG_POLYMORPHIC_IDENTITY}

    def __init__(self, load_file=None, *args, **kwargs):
        self.projection_type = SEEG_POLYMORPHIC_IDENTITY
        self.sensors = sensors.SensorsInternal

        if load_file is not None:
            projection_data = ProjectionSurfaceSEEG.from_file(source_file=load_file)
        else:
            projection_data = None

        super(ProjectionSurfaceSEEG, self).__init__(*args, projection_data=projection_data **kwargs)

    @classmethod
    def from_file(cls, source_file='projection_seeg_588_surface_16k.npy', matlab_data_name=None, is_brainstorm=False,
                  instance=None):
        source_full_path = try_get_absolute_path("tvb_data.projectionMatrix", source_file)
        reader = FileReader(source_full_path)
        if is_brainstorm:
            projection_data = reader.read_gain_from_brainstorm()
        else:
            projection_data = reader.read_array(matlab_data_name=matlab_data_name)
        return projection_data
