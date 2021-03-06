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

The Equation datatypes. This brings together the scientific and framework
methods that are associated with the Equation datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
import json
import numpy
import numexpr
from tvb.basic.traits import core, parameters_factory
from tvb.basic.logger.builder import get_logger


LOG = get_logger(__name__)
# In how many points should the equation be evaluated for the plot. Increasing this will
# give smoother results at the cost of some performance
DEFAULT_PLOT_GRANULARITY = 1024


class Equation(core.Type):
    "Base class for Equation data types."

    # data

    _base_classes = ['Equation', 'FiniteSupportEquation', "DiscreteEquation",
                     "TemporalApplicableEquation", "SpatialApplicableEquation", "HRFKernelEquation",
                     # TODO: There should be a refactor of Coupling which may make these unnecessary
                     'Coupling', 'CouplingData', 'CouplingScientific', 'CouplingFramework',
                     'LinearCoupling', 'LinearCouplingData', 'LinearCouplingScientific', 'LinearCouplingFramework',
                     'SigmoidalCoupling', 'SigmoidalCouplingData', 'SigmoidalCouplingScientific',
                     'SigmoidalCouplingFramework']

    # A latex representation of the equation, with the extra
    # escaping needed for interpretation via sphinx.
    equation = ""

    # sci

    def __init__(self, parameters=None):
        # hackery because if parameters is a dict python points everything to the same dict
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = parameters
        # Should be a list of the parameters and their meaning, Traits
        # should be able to take defaults and sensible ranges from any
        # traited information that was provided.

    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance
        of this datatype.
        """
        summary = {"Equation type": self.__class__.__name__,
                   "equation": self.equation,
                   "parameters": self.parameters}
        return summary

    # ------------------------------ pattern -----------------------------------#
    def _get_pattern(self):
        """
        Return a discrete representation of the equation.
        """
        return self._pattern

    def _set_pattern(self, var):
        """
        Generate a discrete representation of the equation for the space
        represented by ``var``.

        The argument ``var`` can represent a distance, or effective distance,
        for each node in a simulation. Or a time, or in principle any arbitrary
        `` space ``. ``var`` can be a single number, a numpy.ndarray or a
        ?scipy.sparse_matrix? TODO: think this last one is true, need to check
        as we need it for LocalConnectivity...

        """

        self._pattern = numexpr.evaluate(self.equation, global_dict=self.parameters)

    pattern = property(fget=_get_pattern, fset=_set_pattern)

    def get_series_data(self, min_range=0, max_range=100, step=None):
        """
        NOTE: The symbol from the equation which varies should be named: var
        Returns the series data needed for plotting this equation.
        """
        if step is None:
            step = float(max_range - min_range) / DEFAULT_PLOT_GRANULARITY

        var = numpy.arange(min_range, max_range+step, step)
        var = var[numpy.newaxis, :]

        self.pattern = var
        y = self.pattern
        result = zip(var.flat, y.flat)
        return result, False

    @staticmethod
    def build_equation_from_dict(equation_field_name, submitted_data_dict, alter_submitted_dictionary=False):
        """
        Builds from the given data dictionary the equation for the specified field name.
        The dictionary should have the data collapsed.
        """
        if equation_field_name not in submitted_data_dict:
            return None

        eq_param_str = equation_field_name + '_parameters'
        eq = submitted_data_dict.get(eq_param_str)

        equation_parameters = {}
        if eq:
            if 'parameters' in eq:
                equation_parameters = eq['parameters']
            if 'parameters_parameters' in eq:
                equation_parameters = eq['parameters_parameters']

        for k in equation_parameters:
            equation_parameters[k] = float(equation_parameters[k])

        equation_type = submitted_data_dict[equation_field_name]
        equation = parameters_factory.get_traited_instance_for_name(equation_type, Equation,
                                                                    {'parameters': equation_parameters})
        if alter_submitted_dictionary:
            del submitted_data_dict[eq_param_str]
            submitted_data_dict[equation_field_name] = equation

        return equation

    @staticmethod
    def to_json(entity):
        """
        Returns the json representation of this equation.

        The representation of an equation is a dictionary with the following form:
        {'equation_type': '$equation_type', 'parameters': {'$param_name': '$param_value', ...}}
        """
        if entity is not None:
            result = {'__mapped_module': entity.__class__.__module__,
                      '__mapped_class': entity.__class__.__name__,
                      'parameters': entity.parameters}
            return json.dumps(result)
        return None

    @staticmethod
    def from_json(string):
        """
        Retrieves an instance to an equation represented as JSON.

        :param string: the JSON representation of the equation
        :returns: a `tvb.datatypes.equations_data` equation instance
        """
        loaded_dict = json.loads(string)
        if loaded_dict is None:
            return None
        modulename = loaded_dict['__mapped_module']
        classname = loaded_dict['__mapped_class']
        module_entity = __import__(modulename, globals(), locals(), [classname])
        class_entity = getattr(module_entity, classname)
        loaded_instance = class_entity()
        loaded_instance.parameters = loaded_dict['parameters']
        return loaded_instance


class TemporalApplicableEquation(Equation):
    """
    Abstract class introduced just for filtering what equations to be displayed in UI,
    for setting the temporal component in Stimulus on region and surface.
    """
    def __init__(self, *args, **kwargs):
        super(TemporalApplicableEquation, self).__init__(*args, **kwargs)


class FiniteSupportEquation(TemporalApplicableEquation):
    """
    Equations that decay to zero as the variable moves away from zero. It is
    necessary to restrict spatial equation evaluated on a surface to this
    class, are . The main purpose of this class is to facilitate filtering in the UI,
    for patters on surface (stimuli surface and localConnectivity).
    """
    def __init__(self, *args, **kwargs):
        super(FiniteSupportEquation, self).__init__(*args, **kwargs)


class SpatialApplicableEquation(Equation):
    """
    Abstract class introduced just for filtering what equations to be displayed in UI,
    for setting model parameters on the Surface level.
    """
    def __init__(self, *args, **kwargs):
        super(SpatialApplicableEquation, self).__init__(*args, **kwargs)


class DiscreteEquation(FiniteSupportEquation):
    """
    A special case for 'discrete' spaces, such as the regions, where each point
    in the space is effectively just assigned a value.

    """
    # The equation defines a function of :math:`x`
    equation = "var"


class Linear(TemporalApplicableEquation):
    """
    A linear equation.

    """
    # :math:`result = a * x + b`
    equation = "a * var + b"

    def __init__(self, parameters=None, *args, **kwargs):
        if parameters is None:
            params = {"a": 1.0,"b": 0.0}
        else:
            params = parameters
        super(Linear,self).__init__(parameters=params,*args, **kwargs)


class Gaussian(SpatialApplicableEquation, FiniteSupportEquation):
    """
    A Gaussian equation.
    offset: parameter to extend the behaviour of this function
    when spatializing model parameters.

    """

    equation = "(amp * exp(-((var-midpoint)**2 / (2.0 * sigma**2))))+offset"

    def __init__(self, parameters=None, *args, **kwargs):
        if parameters is None:
            params = {"amp": 1.0, "sigma": 1.0, "midpoint": 0.0, "offset": 0.0}
        else:
            params = parameters
        super(Gaussian, self).__init__(parameters=params,*args, **kwargs)


class DoubleGaussian(FiniteSupportEquation):
    """
    A Mexican-hat function approximated by the difference of Gaussians functions.

    """
    _ui_name = "Mexican-hat"

    equation = "(amp_1 * exp(-((var-midpoint_1)**2 / (2.0 * sigma_1**2)))) - (amp_2 * exp(-((var-midpoint_2)**2 / (2.0 * sigma_2**2))))"

    def __init__(self, parameters=None, *args, **kwargs):
        if parameters is None:
            params = {"amp_1": 0.5, "sigma_1": 20.0, "midpoint_1": 0.0,
                      "amp_2": 1.0, "sigma_2": 10.0, "midpoint_2": 0.0}
        else:
            params = parameters
        super(DoubleGaussian, self).__init__(parameters=params, *args, **kwargs)


class Sigmoid(SpatialApplicableEquation, FiniteSupportEquation):
    """
    A Sigmoid equation.
    offset: parameter to extend the behaviour of this function
    when spatializing model parameters.
    """

    equation = "(amp / (1.0 + exp(-1.8137993642342178 * (radius-var)/sigma))) + offset"

    def __init__(self, parameters=None, *args, **kwargs):
        if parameters is None:
            params = {"amp": 1.0, "radius": 5.0, "sigma": 1.0, "offset": 0.0}
        else:
            params = parameters
        super(Sigmoid, self).__init__(parameters=params, *args, **kwargs)


class GeneralizedSigmoid(TemporalApplicableEquation):
    """
    A General Sigmoid equation.
    """

    equation = "low + (high - low) / (1.0 + exp(-1.8137993642342178 * (var-midpoint)/sigma))"

    def __init__(self, parameters=None, *args, **kwargs):
        if parameters is None:
            params = {"low": 0.0, "high": 1.0, "midpoint": 1.0, "sigma": 0.3}
        else:
            params = parameters
        super(GeneralizedSigmoid, self).__init__(parameters=params, *args, **kwargs)


class Sinusoid(TemporalApplicableEquation):
    """
    A Sinusoid equation.
    """

    equation = "amp * sin(6.283185307179586 * frequency * var)"

    def __init__(self, parameters=None, *args, **kwargs):
        if parameters is None:
            params = {"amp": 1.0, "frequency": 0.01}
        else:
            params = parameters
        super(Sinusoid, self).__init__(parameters=params, *args, **kwargs)


class Cosine(TemporalApplicableEquation):
    """
    A Cosine equation.
    """

    equation = "amp * cos(6.283185307179586 * frequency * var)"

    def __init__(self, parameters=None, *args, **kwargs):
        if parameters is None:
            params = {"amp": 1.0, "frequency": 0.01}
        else:
            params = parameters
        super(Cosine, self).__init__(parameters=params, *args,**kwargs)


class Alpha(TemporalApplicableEquation):
    """
    An Alpha function belonging to the Exponential function family.
    """

    equation = "where((var-onset) > 0, (alpha * beta) / (beta - alpha) * (exp(-alpha * (var-onset)) - exp(-beta * (var-onset))), 0.0 * var)"

    def __init__(self, parameters=None, *args, **kwargs):
        if parameters is None:
            params = {"onset": 0.5, "alpha": 13.0, "beta": 42.0}
        else:
            params = parameters
        super(Alpha, self).__init__(parameters=params, *args, **kwargs)


class PulseTrain(TemporalApplicableEquation):
    """
    A pulse train , offset with respect to the time axis.

    **Parameters**:

    * :math:`\\tau` :  pulse width or pulse duration
    * :math:`T`     :  pulse repetition period
    * :math:`f`     :  pulse repetition frequency (1/T)
    * duty cycle    :  :math:``\\frac{\\tau}{T}`` (for a square wave: 0.5)
    * onset time    :
    """

    equation = "where((var % T) < tau, amp, 0)"

    # onset is in milliseconds
    # T and tau are in milliseconds as well

    def __init__(self, parameters=None, *args, **kwargs):
        if parameters is None:
            params = {"T": 42.0, "tau": 13.0, "amp": 1.0, "onset": 30.0}
        else:
            params = parameters
        super(PulseTrain, self).__init__(parameters=params, *args, **kwargs)

    def _get_pattern(self):
        """
        Return a discrete representation of the equation.
        """
        return self._pattern

    def _set_pattern(self, var):
        """
        Generate a discrete representation of the equation for the space
        represented by ``var``.

        The argument ``var`` can represent a distance, or effective distance,
        for each node in a simulation. Or a time, or in principle any arbitrary
        `` space ``. ``var`` can be a single number, a numpy.ndarray or a
        ?scipy.sparse_matrix? TODO: think this last one is true, need to check
        as we need it for LocalConnectivity...

        """

        # rolling in the deep ...
        onset = self.parameters["onset"]
        off = var < onset
        var = numpy.roll(var, off.sum() + 1)
        var[..., off] = 0.0
        self._pattern = numexpr.evaluate(self.equation, global_dict=self.parameters)
        self._pattern[..., off] = 0.0

    pattern = property(fget=_get_pattern, fset=_set_pattern)


class HRFKernelEquation(Equation):
    "Base class for hemodynamic response functions."
    pass


class Gamma(HRFKernelEquation):
    """
    A Gamma function for the bold monitor. It belongs to the family of Exponential functions.

    **Parameters**:


    * :math:`\\tau`      : Exponential time constant of the gamma function [seconds].
    * :math:`n`          : The phase delay of the gamma function.
    * :math: `factorial` : (n-1)!. numexpr does not support factorial yet.
    * :math: `a`         : Amplitude factor after normalization.


    **Reference**:

    .. [B_1996] Geoffrey M. Boynton, Stephen A. Engel, Gary H. Glover and David
        J. Heeger (1996). Linear Systems Analysis of Functional Magnetic Resonance
        Imaging in Human V1. J Neurosci 16: 4207-4221

    .. note:: might be filtered from the equations used in Stimulus and Local Connectivity.

    """

    _ui_name = "HRF kernel: Gamma kernel"

    # TODO: Introduce a time delay in the equation (shifts the hrf onset)
    # """:math:`h(t) = \frac{(\frac{t-\delta}{\tau})^{(n-1)} e^{-(\frac{t-\delta}{\tau})}}{\tau(n-1)!}"""
    # delta = 2.05 seconds -- Additional delay in seconds from the onset of the
    # time-series to the beginning of the gamma hrf.
    # delay cannot be negative or greater than the hrf duration.

    equation = "((var / tau) ** (n - 1) * exp(-(var / tau)) )/ (tau * factorial)"

    def __init__(self, parameters=None, *args, **kwargs):
        if parameters is None:
            params = {"tau": 1.08, "n": 3.0, "factorial": 2.0, "a": 0.1}
        else:
            params = parameters
        super(Gamma, self).__init__(parameters=params, *args, **kwargs)

    def _get_pattern(self):
        """
        Return a discrete representation of the equation.
        """
        return self._pattern

    def _set_pattern(self, var):
        """
        Generate a discrete representation of the equation for the space
        represented by ``var``.

        .. note: numexpr doesn't support factorial yet

        """

        # compute the factorial
        n = int(self.parameters["n"])
        product = 1
        for i in range(n - 1):
            product *= i + 1

        self.parameters["factorial"] = product
        self._pattern = numexpr.evaluate(self.equation,
                                         global_dict=self.parameters)
        self._pattern /= max(self._pattern)
        self._pattern *= self.parameters["a"]

    pattern = property(fget=_get_pattern, fset=_set_pattern)


class DoubleExponential(HRFKernelEquation):
    """
    A difference of two exponential functions to define a kernel for the bold monitor.

    **Parameters** :

    * :math:`\\tau_1`: Time constant of the second exponential function [s]
    * :math:`\\tau_2`: Time constant of the first exponential function [s].
    * :math:`f_1`  : Frequency of the first sine function [Hz].
    * :math:`f_2`  : Frequency of the second sine function [Hz].
    * :math:`amp_1`: Amplitude of the first exponential function.
    * :math:`amp_2`: Amplitude of the second exponential function.
    * :math:`a`    : Amplitude factor after normalization.


    **Reference**:

    .. [P_2000] Alex Polonsky, Randolph Blake, Jochen Braun and David J. Heeger
        (2000). Neuronal activity in human primary visual cortex correlates with
        perception during binocular rivalry. Nature Neuroscience 3: 1153-1159

    """

    _ui_name = "HRF kernel: Difference of Exponentials"

    equation = "((amp_1 * exp(-var/tau_1) * sin(2.*pi*f_1*var)) - (amp_2 * exp(-var/ tau_2) * sin(2.*pi*f_2*var)))"

    def __init__(self, parameters = None, *args, **kwargs):
        if parameters is None:
            params = {"tau_1": 7.22, "f_1": 0.03, "amp_1": 0.1, "tau_2": 7.4,
                      "f_2": 0.12, "amp_2": 0.1, "a": 0.1, "pi": numpy.pi}
        else:
            params = parameters
        super(DoubleExponential, self).__init__(parameters=params, *args, **kwargs)

    def _get_pattern(self):
        """
        Return a discrete representation of the equation.
        """
        return self._pattern

    def _set_pattern(self, var):
        """
        Generate a discrete representation of the equation for the space
        represented by ``var``.

        """

        self._pattern = numexpr.evaluate(self.equation, global_dict=self.parameters)
        self._pattern /= max(self._pattern)

        self._pattern *= self.parameters["a"]

    pattern = property(fget=_get_pattern, fset=_set_pattern)


class FirstOrderVolterra(HRFKernelEquation):
    """
    Integral form of the first Volterra kernel of the three used in the
    Ballon Windekessel model for computing the Bold signal.
    This function describes a damped Oscillator.

    **Parameters** :

    * :math:`\\tau_s`: Dimensionless? exponential decay parameter.
    * :math:`\\tau_f`: Dimensionless? oscillatory parameter.
    * :math:`k_1`    : First Volterra kernel coefficient.
    * :math:`V_0` : Resting blood volume fraction.


    **References** :

    .. [F_2000] Friston, K., Mechelli, A., Turner, R., and Price, C., *Nonlinear
        Responses in fMRI: The Balloon Model, Volterra Kernels, and Other
        Hemodynamics*, NeuroImage, 12, 466 - 477, 2000.

    """

    _ui_name = "HRF kernel: Volterra Kernel"

    equation = "1/3. * exp(-0.5*(var / tau_s)) * (sin(sqrt(1./tau_f - 1./(4.*tau_s**2)) * var)) / (sqrt(1./tau_f - 1./(4.*tau_s**2)))"

    def __init__(self, parameters=None, *args, **kwargs):
        if parameters is None:
            params = {"tau_s": 0.8, "tau_f": 0.4, "k_1": 5.6, "V_0": 0.02}
        else:
            params = parameters
        super(FirstOrderVolterra, self).__init__(parameters=params, *args, **kwargs)


class MixtureOfGammas(HRFKernelEquation):
    """
    A mixture of two gamma distributions to create a kernel similar to the one used in SPM.

    >> import scipy.stats as sp_stats
    >> import numpy
    >> t = numpy.linspace(1,20,100)
    >> a1, a2 = 6., 10.
    >> lambda = 1.
    >> c      = 0.5
    >> hrf    = sp_stats.gamma.pdf(t, a1, lambda) - c * sp_stats.gamma.pdf(t, a2, lambda)

    gamma.pdf(x, a, theta) = (lambda*x)**(a-1) * exp(-lambda*x) / gamma(a)
    a                 : shape parameter
    theta: 1 / lambda : scale parameter


    **References**:

    .. [G_1999] Glover, G. *Deconvolution of Impulse Response in Event-Related BOLD fMRI*.
                NeuroImage 9, 416-429, 1999.


    **Parameters**:


    * :math:`a_{1}`       : shape parameter first gamma pdf.
    * :math:`a_{2}`       : shape parameter second gamma pdf.
    * :math:`\\lambda`    : scale parameter first gamma pdf.


    Default values are based on [G_1999]_:
    * :math:`a_{1} - 1 = n_{1} =  5.0`
    * :math:`a_{2} - 1 = n_{2} = 12.0`
    * :math:`c \\equiv a_{2}   = 0.4`

    Alternative values :math:`a_{2}=10` and :math:`c=0.5`

    NOTE: gamma_a_1 and gamma_a_2 are placeholders, the true values are
    computed before evaluating the expression, because numexpr does not
    support certain functions.

    NOTE: [G_1999]_ used a different analytical function that can be approximated
    by this difference of gamma pdfs

    """

    _ui_name = "HRF kernel: Mixture of Gammas"

    equation = "(l * var)**(a_1-1) * exp(-l*var) / gamma_a_1 - c * (l*var)**(a_2-1) * exp(-l*var) / gamma_a_2"



    def __init__(self, parameters=None, *args, **kwargs):
        if parameters is None:
            params = {"a_1": 6.0, "a_2": 13.0, "l": 1.0, "c": 0.4,
                      "gamma_a_1": 1.0, "gamma_a_2": 1.0}
        else:
            params = parameters
        super(MixtureOfGammas, self).__init__(parameters=params*args, **kwargs)

    def _get_pattern(self):
        """
        Return a discrete representation of the equation.
        """
        return self._pattern

    def _set_pattern(self, var):
        """
        Generate a discrete representation of the equation for the space
        represented by ``var``.

        .. note: numexpr doesn't support gamma function

        """

        # get gamma functions
        from scipy.special import gamma as sp_gamma
        self.parameters["gamma_a_1"] = sp_gamma(self.parameters["a_1"])
        self.parameters["gamma_a_2"] = sp_gamma(self.parameters["a_2"])

        self._pattern = numexpr.evaluate(self.equation, global_dict=self.parameters)

    pattern = property(fget=_get_pattern, fset=_set_pattern)
