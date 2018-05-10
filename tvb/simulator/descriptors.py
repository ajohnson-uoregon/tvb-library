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

"""
Data descriptors for declaring workspace for algorithms and checking usage.

.. moduleauthor:: Marmaduke Woodman <mmwoodman@gmail.com>

"""

import numpy
import collections
import weakref
import six
from .common import get_logger

LOG = get_logger(__name__)

# StaticAttr prevents descriptors from placing instance-owned, descriptor storage


class Final(object):
    "A descriptor for an attribute, possibly type-checked, that once initialized, cannot be changed."

    State = collections.namedtuple('State', 'value initialized')

    def __init__(self, type=None):
        self.instance_state = weakref.WeakKeyDictionary()
        self.type = type

    def _get_or_create_state(self, instance):
        if instance not in self.instance_state:
            self.instance_state[instance] = Final.State(None, False)
        return self.instance_state[instance]

    def _correct_type(self, value):
        return isinstance(value, self.type)

    def __set__(self, instance, value):
        state = self._get_or_create_state(instance) # type: Final.State
        if state.initialized:
            raise AttributeError('final attribute cannot be set.')
        else:
            if self.type and not self._correct_type(value):
                raise AttributeError('value %r does not match expected type %r'
                                      % (value, self.type))
            self.instance_state[instance] = Final.State(value, True)

    def __get__(self, instance, owner):
        if instance is None:
            LOG.debug('Final returning self for None instance.')
            return self
        else:
            return self._get_or_create_state(instance).value
