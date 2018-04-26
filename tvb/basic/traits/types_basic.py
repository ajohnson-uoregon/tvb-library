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
This module describes the simple of traited attributes one might needs on a class.

If your subclass should be mapped to a database table (true for most entities
that will be reused), use MappedType as superclass.

If you subclass is supported natively by SQLAlchemy, subclass Type, otherwise
subclass MappedType.

Important:
- Type - traited, possible mapped to db *col*
- MappedType - traited, mapped to db *table*


.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: marmaduke <duke@eml.cc>
.. moduleauthor:: Paula Sanz-Leon <paula.sanz-leon@univ-amu.fr>
"""

import json
import numpy
from decimal import Decimal
import tvb.basic.traits.core as core
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)


class String(core.Type):
    """
    Traits type that wraps a Python string.
    """
    wraps = (str, unicode)


class MapAsJson():
    """Add functionality of converting from/to JSON"""


    def __get__(self, inst, cls):
        if inst is not None and self.trait.bound and hasattr(inst, '_' + self.trait.name):

            if hasattr(inst, '__' + self.trait.name):
                return getattr(inst, '__' + self.trait.name)

            string = getattr(inst, '_' + self.trait.name)
            if string is None or (not isinstance(string, (str, unicode))):
                return string
            if len(string) < 1:
                return None
            json_value = self.from_json(string)

            # Cache for future usages (e.g. Stimulus.spatial should be the same instance on multiple accesses)
            setattr(inst, '__' + self.trait.name, json_value)
            return json_value
        return self


    @staticmethod
    def to_json(entity):
        return json.dumps(entity)


    @staticmethod
    def from_json(string):
        return json.loads(string)


    @staticmethod
    def decode_map_as_json(dct):
        """
        Used in the __convert_to_array to get an equation from the UI corresponding string.
        """
        for key, value in dct.items():
            if isinstance(value, unicode) and '__mapped_module' in value:
                dict_value = json.loads(value)
                if '__mapped_module' not in dict_value:
                    dct[key] = MapAsJson.decode_map_as_json(dict_value)
                else:
                    modulename = dict_value['__mapped_module']
                    classname = dict_value['__mapped_class']
                    module_entity = __import__(modulename, globals(), locals(), [classname])
                    class_entity = eval('module_entity.' + classname)
                    loaded_entity = class_entity.from_json(value)
                    dct[key] = loaded_entity
        return dct

    class MapAsJsonEncoder(json.JSONEncoder):
        """
        Used before any save to the database to encode Equation type objects.
        """


        def default(self, obj):
            if isinstance(obj, MapAsJson):
                return obj.to_json(obj)
            else:
                return json.JSONEncoder.default(self, obj)


class JSONType(String):
    """
    Wrapper over a String which holds a serializable object.
    On set/get JSON load/dump will be called.
    """


    def __get__(self, inst, cls):
        if inst:
            string = super(JSONType, self).__get__(inst, cls)
            if string is None or (not isinstance(string, (str, unicode))):
                return string
            if len(string) < 1:
                return None
            return json.loads(string)
        return super(JSONType, self).__get__(inst, cls)


    def __set__(self, inst, value):
        if not isinstance(value, (str, unicode)):
            value = json.dumps(value)
        super(JSONType, self).__set__(inst, value)


class DType(String):
    """
    Traits type that wraps a Numpy dType specification.
    """

    wraps = (numpy.dtype, str)
    defaults = ((numpy.float64,), {})


    def __get__(self, inst, cls):
        if inst:
            type_ = super(DType, self).__get__(inst, cls)
            return str(type_).replace("<type '", '').replace("'>", '')
        return super(DType, self).__get__(inst, cls)


    def __set__(self, inst, value):
        super(DType, self).__set__(inst, str(value))
