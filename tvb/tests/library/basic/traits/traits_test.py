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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""
import numpy
import json
from copy import deepcopy
from tvb.tests.library.base_testcase import BaseTestCase
import tvb.datatypes.equations as equations
import tvb.datatypes.time_series as time_series
from tvb.simulator.models import WilsonCowan, ReducedSetHindmarshRose


class TestTraits(BaseTestCase):
    """
    Test class for traits.core and traits.base
    """

    def test_default_attributes(self):
        """
        Test that default attributes are populated as they are described in Traits.
        """
        model_wc = WilsonCowan()
        assert type(model_wc.c_ii) == numpy.ndarray
        assert 1 == len(model_wc.c_ii)
        assert type(model_wc.tau_e) == numpy.ndarray
        assert 1 == len(model_wc.tau_e)

    def test_modifying_attributes(self):
        """
        Test that when modifying an instance attributes, they are only visible on that instance.
        """
        eqn_t = equations.Gaussian()
        eqn_t.parameters["midpoint"] = 8.0

        eqn_x = equations.Gaussian()
        eqn_x.parameters["amp"] = -0.0625

        assert eqn_t.parameters is not None
        assert eqn_t.parameters['amp'] == 1.0
        assert eqn_t.parameters['midpoint'] == 8.0
        assert eqn_x.parameters is not None
        assert eqn_x.parameters['amp'] == -0.0625
        assert eqn_x.parameters['midpoint'] == 0.0

        a1 = numpy.array([20.0, ])
        a2 = deepcopy(a1)
        a2 = numpy.array([42.0, ])
        assert a1[0] != a2[0]

        model1 = ReducedSetHindmarshRose()
        model1.a = numpy.array([55.0, ])
        model1.number_of_modes = 10

        model2 = ReducedSetHindmarshRose()
        model2.a = numpy.array([42.0])
        model2.number_of_modes = 15
        assert model1.number_of_modes != model2.number_of_modes
        assert model1.a[0] != model2.a[0]

    def test_linked_attributes(self):
        """
        Test that tvb.core.trait.util produces something.
        """

        class Internal_Class(object):
            """ Dummy persisted class"""
            x = 5
            z = 0
            j = {}

            class In_Internal_Class(object):
                """Internal of Internal class"""
                t = numpy.array(range(10))

            @property
            def y(self):
                return self.x

        instance = Internal_Class()
        assert 5 == instance.y

        instance.j = {'dict_key': 10}
        assert isinstance(instance.j, dict)

    def test_none_complex_attribute(self):
        """
        Test that traited attributes are returned None,
        when no value is assigned as default to them.
        """
        serie = time_series.TimeSeriesRegion()
        assert serie.connectivity is None
