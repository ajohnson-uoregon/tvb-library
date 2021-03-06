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
#   The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
A collection of noise related classes and functions.

Specific noises inherit from the abstract class Noise, with each instance having
its own RandomStream attribute -- which is itself a Traited wrapper of Numpy's
RandomState.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>
.. moduleauthor:: Noelia Montejo <Noelia@tvb.invalid>

"""

import numpy
from tvb.datatypes import equations
from .common import get_logger, simple_gen_astr


LOG = get_logger(__name__)


class Noise(object):
    """
    Defines a base class for noise. Specific noises are derived from this class
    for use in stochastic integrations.

    .. [KloedenPlaten_1995] Kloeden and Platen, Springer 1995, *Numerical
        solution of stochastic differential equations.*

    .. [ManellaPalleschi_1989] Manella, R. and Palleschi V., *Fast and precise
        algorithm for computer simulation of stochastic differential equations*,
        Physical Review A, Vol. 40, Number 6, 1989. [3381-3385]

    .. [Mannella_2002] Mannella, R.,  *Integration of Stochastic Differential
        Equations on a Computer*, Int J. of Modern Physics C 13(9): 1177--1194,
        2002.

    .. [FoxVemuri_1988] Fox, R., Gatland, I., Rot, R. and Vemuri, G., * Fast ,
        accurate algorithm for simulation of exponentially correlated colored
        noise*, Physical Review A, Vol. 38, Number 11, 1988. [5938-5940]


    .. #Currently there seems to be a clash betwen traits and autodoc, autodoc
    .. #can't find the methods of the class, the class specific names below get
    .. #us around this...
    .. automethod:: Noise.__init__
    .. automethod:: Noise.configure_white
    .. automethod:: Noise.generate
    .. automethod:: Noise.white
    .. automethod:: Noise.coloured

    """
    _base_classes = ['Noise', 'MultiplicativeSimple']

    #NOTE: nsig is not declared here because we use this class directly as the
    #      inital conditions noise source, and in that use the job of nsig is
    #      filled by the state_variable_range attribute of the Model.


    def __init__(self, ntau=0.0, dt=None, _E=None, _sqrt_1_E2=None, _eta=None,
                 _h=None, random_stream=None):
        self.ntau = ntau
        self.dt = dt
        # for use with colored noise
        self._E = _E
        self._sqrt_1_E2 = _sqrt_1_E2
        self._eta = _eta
        self._h = _h

        if random_stream is None:
            random_stream = numpy.random.RandomState()
        self.random_stream = random_stream

    def configure(self):
        """
        Run base classes configure to setup traited attributes, then ensure that
        the ``random_stream`` attribute is properly configured.

        """
        super(Noise, self).configure()

    def __str__(self):
        return simple_gen_astr(self, 'dt ntau')

    def configure_white(self, dt, shape=None):
        """Set the time step (dt) of noise or integration time"""
        self.dt = dt
        LOG.info('White noise configured with dt=%g', self.dt)

    def configure_coloured(self, dt, shape):
        r"""
        One of the simplest forms for coloured noise is exponentially correlated
        Gaussian noise [KloedenPlaten_1995]_.

        We give the initial conditions for coloured noise using the integral
        algorith for simulating exponentially correlated noise proposed by
        [FoxVemuri_1988]_

        To start the simulation, an initial value for :math:`\eta` is needed.
        It is obtained in accord with Eqs.[13-15]:

            .. math::
                m &= \text{random number}\\
                n &= \text{random number}\\
                \eta &= \sqrt{-2D\lambda\ln(m)}\,\cos(2\pi\,n)

        where :math:`D` is standard deviation of the noise amplitude and
        :math:`\lambda = \frac{1}{\tau_n}` is the inverse of the noise
        correlation time. Then we set :math:`E = \exp{-\lambda\,\delta\,t}`
        where :math:`\delta\,t` is the integration time step.

        After that the exponentially correlated, coloured noise, is obtained:

            .. math::
                a &= \text{random number}\\
                b &= \text{random number}\\
                h &= \sqrt{-2D\lambda\,(1 - E^2)\,\ln{a}}\,\cos(2\pi\,b)\\
                \eta_{t+\delta\,t} &= \eta_{t}E + h

        """
        #TODO: Probably best to change the docstring to be consistent with the
        #      below, ie, factoring out the explicit Box-Muller.
        #NOTE: The actual implementation factors out the explicit Box-Muller,
        #      using numpy's normal() instead.
        self.dt = dt
        self._E = numpy.exp(-self.dt / self.ntau)
        self._sqrt_1_E2 = numpy.sqrt((1.0 - self._E ** 2))
        self._eta = self.random_stream.normal(size=shape)
        self._dt_sqrt_lambda = self.dt * numpy.sqrt(1.0 / self.ntau)
        LOG.info('Colored noise configured with dt=%g E=%g sqrt_1_E2=%g eta=%g & dt_sqrt_lambda=%g',
                  self.dt, self._E, self._sqrt_1_E2, self._eta, self._dt_sqrt_lambda)

    def generate(self, shape, lo=-1.0, hi=1.0):
        "Generate noise realization."
        if self.ntau > 0.0:
            noise = self.coloured(shape)
        else:
            noise = self.white(shape)
        return noise

    def coloured(self, shape):
        "Generate colored noise. [FoxVemuri_1988]_"
        self._h = self._sqrt_1_E2 * self.random_stream.normal(size=shape)
        self._eta =  self._eta * self._E + self._h
        return self._dt_sqrt_lambda * self._eta

    def white(self, shape):
        "Generate white noise."
        noise = numpy.sqrt(self.dt) * self.random_stream.normal(size=shape)
        return noise


class Additive(Noise):
    """
    Additive noise which, assuming the source noise is Gaussian with unit
    variance, will result in noise with a standard deviation of nsig.

    """

    # The noise dispersion, it is the standard deviation of the
    # distribution from which the Gaussian random variates are drawn. NOTE:
    # Sensible values are typically ~<< 1% of the dynamic range of a Model's
    # state variables.


    def __init__(self, nsig=None, *args, **kwargs):
        if nsig is None:
            nsig = numpy.array([1.0], dtype=numpy.float64)
        self.nsig = nsig

        super(Additive, self).__init__(*args, **kwargs)

    def gfun(self, state_variables):
        r"""
        Linear additive noise, thus it ignores the state_variables.

        .. math::
            g(x) = \sqrt{2D}

        """
        g_x = numpy.sqrt(2.0 * self.nsig)
        return g_x


class Multiplicative(Noise):
    r"""
    With "external" fluctuations the intensity of the noise often depends on
    the state of the system. This results in the (general) stochastic
    differential formulation:

    .. math::
        dX_t = a(X_t)\,dt + b(X_t)\,dW_t

    for appropriate coefficients :math:`a(x)` and :math:`b(x)`, which might be
    constants.

    From [KloedenPlaten_1995]_, Equation 1.9, page 104.

    """


    def __init__(self, nsig=None, b=None, *args, **kwargs):
        if nsig is None:
            # The noise dispersion, it is the standard deviation of the
            # distribution from which the Gaussian random variates are drawn. NOTE:
            # Sensible values are typically ~<< 1% of the dynamic range of a Model's
            # state variables.
            nsig = numpy.array([1.0], dtype=numpy.float64)
        self.nsig = nsig

        if b is None:
            b = equations.Linear(parameters={"a": 1.0, "b": 0.0})
        self.b = b

        super(Multiplicative, self).__init__(*args, **kwargs)

    def gfun(self, state_variables):
        """
        Scale the noise by the noise dispersion and the diffusion coefficient.
        By default, the diffusion coefficient :math:`b` is a constant.
        It reduces to the simplest scheme of a linear SDE with Multiplicative
        Noise: homogeneous constant coefficients. See [KloedenPlaten_1995]_,
        Equation 4.6, page 119.

        """
        self.b.pattern = state_variables
        g_x = numpy.sqrt(2.0 * self.nsig) * self.b.pattern
        return g_x
