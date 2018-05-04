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
The Mode Decomposition datatypes. This brings together the scientific and
framework methods that are associated with the Mode Decomposition datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <paula.sanz-leon@univ-amu.fr>

"""

import numpy
from tvb.basic.logger.builder import get_logger
import tvb.basic.traits.core as core
import tvb.datatypes.arrays as arrays
import tvb.datatypes.time_series as time_series
from tvb.basic.traits.types_mapped import MappedType


LOG = get_logger(__name__)


class PrincipalComponents(MappedType):
    """
    Result of a Principal Component Analysis (PCA).
    """

    source = time_series.TimeSeries(
        label="Source time-series",
        doc="Links to the time-series on which the PCA is applied.")


    def __init__(self, weights=None, fractions=None, norm_source=None,
                 component_time_series=None, normalised_component_time_series=None,
                 *args, **kwargs):
        if weights is None:
            weights = numpy.array([])
        self.weights = weights

        if fractions is None:
            fractions = numpy.array([])
        self.fractions = fractions

        if norm_source is None:
            norm_source = numpy.array([])
        self.norm_source = norm_source

        if component_time_series is None:
            component_time_series = numpy.array([])
        self.component_time_series = component_time_series

        if normalised_component_time_series is None:
            normalised_component_time_series = numpy.array([])
        self.normalised_component_time_series = normalised_component_time_series

        super(PrincipalComponents, self).__init__(*args, **kwargs)

    def write_data_slice(self, partial_result):
        """
        Append chunk.
        """
        self.store_data_chunk('weights', partial_result.weights, grow_dimension=2, close_file=False)

        self.store_data_chunk('fractions', partial_result.fractions, grow_dimension=1, close_file=False)

        partial_result.compute_norm_source()
        self.store_data_chunk('norm_source', partial_result.norm_source, grow_dimension=1, close_file=False)

        partial_result.compute_component_time_series()
        self.store_data_chunk('component_time_series', partial_result.component_time_series,
                              grow_dimension=1, close_file=False)

        partial_result.compute_normalised_component_time_series()
        self.store_data_chunk('normalised_component_time_series', partial_result.normalised_component_time_series,
                              grow_dimension=1, close_file=False)

    def read_fractions_data(self, from_comp, to_comp):
        """
        Return a list with fractions for components in interval from_comp, to_comp and in
        addition have in position n the sum of the fractions for the rest of the components.
        """
        from_comp = int(from_comp)
        to_comp = int(to_comp)
        all_data = self.get_data('fractions').flat
        sum_others = 0
        for idx, val in enumerate(all_data):
            if idx < from_comp or idx > to_comp:
                sum_others += val
        return numpy.array(all_data[from_comp:to_comp].tolist() + [sum_others])

    def read_weights_data(self, from_comp, to_comp):
        """
        Return the weights data for the components in the interval [from_comp, to_comp].
        """
        from_comp = int(from_comp)
        to_comp = int(to_comp)
        data_slice = slice(from_comp, to_comp, None)
        weights_shape = self.get_data_shape('weights')
        weights_slice = [slice(size) for size in weights_shape]
        weights_slice[0] = data_slice
        weights_data = self.get_data('weights', tuple(weights_slice))
        return weights_data.flatten()

    def configure(self):
        """
        Invoke the compute methods for computable attributes that haven't been
        set during initialization.
        """
        super(PrincipalComponents, self).configure()

        if self.trait.use_storage is False and sum(self.get_data_shape('weights')) != 0:
            if self.norm_source.size == 0:
                self.compute_norm_source()

            if self.component_time_series.size == 0:
                self.compute_component_time_series()

            if self.normalised_component_time_series.size == 0:
                self.compute_normalised_component_time_series()

    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance
        of this datatype.
        """
        summary = {"Mode decomposition type": self.__class__.__name__}
        summary["Source"] = self.source.title
        # summary["Number of variables"] = self...
        # summary["Number of mewasurements"] = self...
        # summary["Number of components"] = self...
        # summary["Number required for 95%"] = self...
        return summary

    def compute_norm_source(self):
        """Normalised source time-series."""
        self.norm_source = ((self.source.data - self.source.data.mean(axis=0)) /
                            self.source.data.std(axis=0))

    # TODO: ??? Any value in making this a TimeSeries datatypes ???
    def compute_component_time_series(self):
        """Compnent time-series."""
        # TODO: Generalise -- it currently assumes 4D TimeSeriesSimulator...
        ts_shape = self.source.data.shape
        component_ts = numpy.zeros(ts_shape)
        for var in range(ts_shape[1]):
            for mode in range(ts_shape[3]):
                w = self.weights[:, :, var, mode]
                ts = self.source.data[:, var, :, mode]
                component_ts[:, var, :, mode] = numpy.dot(w, ts.T).T

        self.component_time_series = component_ts

    # TODO: ??? Any value in making this a TimeSeries datatypes ???
    def compute_normalised_component_time_series(self):
        """normalised_Compnent time-series."""
        # TODO: Generalise -- it currently assumes 4D TimeSeriesSimulator...
        ts_shape = self.source.data.shape
        component_ts = numpy.zeros(ts_shape)
        for var in range(ts_shape[1]):
            for mode in range(ts_shape[3]):
                w = self.weights[:, :, var, mode]
                nts = self.norm_source[:, var, :, mode]
                component_ts[:, var, :, mode] = numpy.dot(w, nts.T).T

        self.normalised_component_time_series = component_ts


class IndependentComponents(MappedType):
    """
    Result of an Independent Component Analysis.

    """
    source = time_series.TimeSeries(
        label="Source time-series",
        doc="Links to the time-series on which the ICA is applied.")

    def __init__(self, n_components=0, mixing_matrix=None, unmixing_matrix=None,
                 prewhitening_matrix=None, norm_source=None,
                 component_time_series=None, normalised_component_time_series=None,
                 *args, **kwargs):
        self.n_components = n_components # Number of independent components
        # Observed data matrix is considered to be a linear combination
        # of n non-Gaussian independent components

        if mixing_matrix is None:
            mixing_matrix = numpy.array([])
        self.mixing_matrix = mixing_matrix

        if unmixing_matrix is None:
            unmixing_matrix = numpy.array([])
        self.unmixing_matrix = unmixing_matrix

        if prewhitening_matrix is None:
            prewhitening_matrix = numpy.array([])
        self.prewhitening_matrix = prewhitening_matrix

        if norm_source is None:
            norm_source = numpy.array([])
        self.norm_source = norm_source

        if component_time_series is None:
            component_time_series = numpy.array([])
        self.component_time_series = component_time_series

        if normalised_component_time_series is None:
            normalised_component_time_series = numpy.array([])
        self.normalised_component_time_series = normalised_component_time_series

        super(IndependentComponents, self).__init__(*args, **kwargs)

    def write_data_slice(self, partial_result):
        """
        Append chunk.

        """
        self.store_data_chunk('unmixing_matrix', partial_result.unmixing_matrix, grow_dimension=2, close_file=False)
        self.store_data_chunk('prewhitening_matrix', partial_result.prewhitening_matrix,
                              grow_dimension=2, close_file=False)
        partial_result.compute_norm_source()
        self.store_data_chunk('norm_source', partial_result.norm_source, grow_dimension=1, close_file=False)
        partial_result.compute_component_time_series()
        self.store_data_chunk('component_time_series', partial_result.component_time_series,
                              grow_dimension=1, close_file=False)
        partial_result.compute_normalised_component_time_series()
        self.store_data_chunk('normalised_component_time_series', partial_result.normalised_component_time_series,
                              grow_dimension=1, close_file=False)
        partial_result.compute_mixing_matrix()
        self.store_data_chunk('mixing_matrix', partial_result.mixing_matrix, grow_dimension=2, close_file=False)

    def configure(self):
        """
        Invoke the compute methods for computable attributes that haven't been
        set during initialisation.
        """
        super(IndependentComponents, self).configure()
        if self.trait.use_storage is False and sum(self.get_data_shape('unmixing_matrix')) != 0:
            if self.norm_source.size == 0:
                self.compute_norm_source()
            if self.component_time_series.size == 0:
                self.compute_component_time_series()
            if self.normalised_component_time_series.size == 0:
                self.compute_normalised_component_time_series()

    def compute_norm_source(self):
        """Normalised source time-series."""
        self.norm_source = ((self.source.data - self.source.data.mean(axis=0)) /
                            self.source.data.std(axis=0))

    def compute_component_time_series(self):
        ts_shape = self.source.data.shape
        component_ts_shape = (ts_shape[0], ts_shape[1], self.n_components, ts_shape[3])
        component_ts = numpy.zeros(component_ts_shape)
        for var in range(ts_shape[1]):
            for mode in range(ts_shape[3]):
                w = self.unmixing_matrix[:, :, var, mode]
                k = self.prewhitening_matrix[:, :, var, mode]
                ts = self.source.data[:, var, :, mode]
                component_ts[:, var, :, mode] = numpy.dot(w, numpy.dot(k, ts.T)).T
        self.component_time_series = component_ts

    def compute_normalised_component_time_series(self):
        ts_shape = self.source.data.shape
        component_ts_shape = (ts_shape[0], ts_shape[1], self.n_components, ts_shape[3])
        component_nts = numpy.zeros(component_ts_shape)
        for var in range(ts_shape[1]):
            for mode in range(ts_shape[3]):
                w = self.unmixing_matrix[:, :, var, mode]
                k = self.prewhitening_matrix[:, :, var, mode]
                nts = self.norm_source[:, var, :, mode]
                component_nts[:, var, :, mode] = numpy.dot(w, numpy.dot(k, nts.T)).T
        self.normalised_component_time_series = component_nts

    def compute_mixing_matrix(self):
        """
        Compute the linear mixing matrix A, so X = A * S ,
        where X is the observed data and S contain the independent components
            """
        ts_shape = self.source.data.shape
        mixing_matrix_shape = (ts_shape[2], self.n_components, ts_shape[1], ts_shape[3])
        mixing_matrix = numpy.zeros(mixing_matrix_shape)
        for var in range(ts_shape[1]):
            for mode in range(ts_shape[3]):
                w = self.unmixing_matrix[:, :, var, mode]
                k = self.prewhitening_matrix[:, :, var, mode]
                temp = numpy.matrix(numpy.dot(w, k))
                mixing_matrix[:, :, var, mode] = numpy.array(numpy.dot(temp.T, (numpy.dot(temp, temp.T)).T))
        self.mixing_matrix = mixing_matrix

    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance
        of this datatype.
        """
        summary = {"Mode decomposition type": self.__class__.__name__}
        summary["Source"] = self.source.title
        return summary
