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

"""
Jansen-Rit and derivative models.

"""

from .base import ModelNumbaDfun, Model, numpy
import math
from numba import guvectorize, float64
from enum import Enum, unique

class JansenRit(ModelNumbaDfun):
    r"""
    The Jansen and Rit is a biologically inspired mathematical framework
    originally conceived to simulate the spontaneous electrical activity of
    neuronal assemblies, with a particular focus on alpha activity, for instance,
    as measured by EEG. Later on, it was discovered that in addition to alpha
    activity, this model was also able to simulate evoked potentials.

    .. [JR_1995]  Jansen, B., H. and Rit V., G., *Electroencephalogram and
        visual evoked potential generation in a mathematical model of
        coupled cortical columns*, Biological Cybernetics (73) 357:366, 1995.

    .. [J_1993] Jansen, B., Zouridakis, G. and Brandt, M., *A
        neurophysiologically-based mathematical model of flash visual evoked
        potentials*

    .. figure :: img/JansenRit_45_mode_0_pplane.svg
        :alt: Jansen and Rit phase plane (y4, y5)

        The (:math:`y_4`, :math:`y_5`) phase-plane for the Jansen and Rit model.

    .. automethod:: JansenRit.__init__

    The dynamic equations were taken from [JR_1995]_

    .. math::
        \dot{y_0} &= y_3 \\
        \dot{y_3} &= A a\,S[y_1 - y_2] - 2a\,y_3 - 2a^2\, y_0 \\
        \dot{y_1} &= y_4\\
        \dot{y_4} &= A a \,[p(t) + \alpha_2 J + S[\alpha_1 J\,y_0]+ c_0]
                    -2a\,y - a^2\,y_1 \\
        \dot{y_2} &= y_5 \\
        \dot{y_5} &= B b (\alpha_4 J\, S[\alpha_3 J \,y_0]) - 2 b\, y_5
                    - b^2\,y_2 \\
        S[v] &= \frac{2\, \nu_{max}}{1 + \exp^{r(v_0 - v)}}

    """

    _ui_name = "Jansen-Rit"
    ui_configurable_parameters = ['A', 'B', 'a', 'b', 'v0', 'nu_max', 'r', 'J',
                                  'a_1', 'a_2', 'a_3', 'a_4', 'p_min', 'p_max',
                                  'mu']

    # Define traited attributes for this model, these represent possible kwargs.
    A = numpy.array([3.25])

    B = numpy.array([22.0])

    a = numpy.array([0.1])

    b = numpy.array([0.05])

    v0 = numpy.array([5.52])

    nu_max = numpy.array([0.0025])

    r = numpy.array([0.56])

    J = numpy.array([135.0])

    a_1 = numpy.array([1.0])

    a_2 = numpy.array([0.8])

    a_3 = numpy.array([0.25])

    a_4 = numpy.array([0.25])

    p_min = numpy.array([0.12])

    p_max = numpy.array([0.32])

    mu = numpy.array([0.22])

    #Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = {"y0": numpy.array([-1.0, 1.0]),
                            "y1": numpy.array([-500.0, 500.0]),
                            "y2": numpy.array([-50.0, 50.0]),
                            "y3": numpy.array([-6.0, 6.0]),
                            "y4": numpy.array([-20.0, 20.0]),
                            "y5": numpy.array([-500.0, 500.0])}
    # The values for each state-variable should be set to encompass
    # the expected dynamic range of that state-variable for the current
    # parameters, it is used as a mechanism for bounding random inital
    # conditions when the simulation isn't started from an explicit history,
    # it is also provides the default range of phase-plane plots

    @unique
    class Variables(Enum):
        Y0 = "y0"
        Y1 = "y1"
        Y2 = "y2"
        Y3 = "y3"
        Y4 = "y4"
        Y5 = "y5"

        def __get__(self, obj, type):
            return self.value




    state_variables = 'y0 y1 y2 y3 y4 y5'.split()
    _nvar = 6
    cvar = numpy.array([1, 2], dtype=numpy.int32)

    def __init__(self, variables_of_interest=None, *args, **kwargs):
        if variables_of_interest is None:
            variables_of_interest = [self.Variables.Y0, self.Variables.Y1, self.Variables.Y2, self.Variables.Y3]
        super(JansenRit, self).__init__(variables_of_interest=variables_of_interest, *args, **kwargs)

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The dynamic equations were taken from [JR_1995]_

        .. math::
            \dot{y_0} &= y_3 \\
            \dot{y_3} &= A a\,S[y_1 - y_2] - 2a\,y_3 - 2a^2\, y_0 \\
            \dot{y_1} &= y_4\\
            \dot{y_4} &= A a \,[p(t) + \alpha_2 J S[\alpha_1 J\,y_0]+ c_0]
                        -2a\,y - a^2\,y_1 \\
            \dot{y_2} &= y_5 \\
            \dot{y_5} &= B b (\alpha_4 J\, S[\alpha_3 J \,y_0]) - 2 b\, y_5
                        - b^2\,y_2 \\
            S[v] &= \frac{2\, \nu_{max}}{1 + \exp^{r(v_0 - v)}}


        :math:`p(t)` can be any arbitrary function, including white noise or
        random numbers taken from a uniform distribution, representing a pulse
        density with an amplitude varying between 120 and 320

        For Evoked Potentials, a transient component of the input,
        representing the impulse density attribuable to a brief visual input is
        applied. Time should be in seconds.

        .. math::
            p(t) = q\,(\frac{t}{w})^n \, \exp{-\frac{t}{w}} \\
            q = 0.5 \\
            n = 7 \\
            w = 0.005 [s]

        """
        y0, y1, y2, y3, y4, y5 = state_variables

        # NOTE: This is assumed to be \sum_j u_kj * S[y_{1_j} - y_{2_j}]
        lrc = coupling[0, :]
        short_range_coupling =  local_coupling*(y1 -  y2)

        # NOTE: for local couplings
        # 0: pyramidal cells
        # 1: excitatory interneurons
        # 2: inhibitory interneurons
        # 0 -> 1,
        # 0 -> 2,
        # 1 -> 0,
        # 2 -> 0,

        exp = numpy.exp
        sigm_y1_y2 = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - (y1 - y2))))
        sigm_y0_1  = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - (self.a_1 * self.J * y0))))
        sigm_y0_3  = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - (self.a_3 * self.J * y0))))

        return numpy.array([
            y3,
            y4,
            y5,
            self.A * self.a * sigm_y1_y2 - 2.0 * self.a * y3 - self.a ** 2 * y0,
            self.A * self.a * (self.mu + self.a_2 * self.J * sigm_y0_1 + lrc + short_range_coupling)
                - 2.0 * self.a * y4 - self.a ** 2 * y1,
            self.B * self.b * (self.a_4 * self.J * sigm_y0_3) - 2.0 * self.b * y5 - self.b ** 2 * y2,
        ])

    def dfun(self, y, c, local_coupling=0.0):
        src =  local_coupling*(y[1] - y[2])[:, 0]
        y_ = y.reshape(y.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_jr(y_, c_, src,
                               self.nu_max, self.r, self.v0, self.a, self.a_1, self.a_2, self.a_3, self.a_4,
                               self.A, self.b, self.B, self.J, self.mu
                               )
        return deriv.T[..., numpy.newaxis]


@guvectorize([(float64[:],) * 17], '(n),(m)' + ',()'*14 + '->(n)', nopython=True)
def _numba_dfun_jr(y, c,
                   src,
                   nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu,
                   dx):
    sigm_y1_y2 = 2.0 * nu_max[0] / (1.0 + math.exp(r[0] * (v0[0] - (y[1] - y[2]))))
    sigm_y0_1 = 2.0 * nu_max[0] / (1.0 + math.exp(r[0] * (v0[0] - (a_1[0] * J[0] * y[0]))))
    sigm_y0_3 = 2.0 * nu_max[0] / (1.0 + math.exp(r[0] * (v0[0] - (a_3[0] * J[0] * y[0]))))
    dx[0] = y[3]
    dx[1] = y[4]
    dx[2] = y[5]
    dx[3] = A[0] * a[0] * sigm_y1_y2 - 2.0 * a[0] * y[3] - a[0] ** 2 * y[0]
    dx[4] = A[0] * a[0] * (mu[0] + a_2[0] * J[0] * sigm_y0_1 + c[0] + src[0]) - 2.0 * a[0] * y[4] - a[0] ** 2 * y[1]
    dx[5] = B[0] * b[0] * (a_4[0] * J[0] * sigm_y0_3) - 2.0 * b[0] * y[5] - b[0] ** 2 * y[2]


class ZetterbergJansen(Model):
    """
    Zetterberg et al derived a model inspired by the Wilson-Cowan equations. It served as a basis for the later,
    better known Jansen-Rit model.

    .. [ZL_1978] Zetterberg LH, Kristiansson L and Mossberg K. Performance of a Model for a Local Neuron Population.
        Biological Cybernetics 31, 15-26, 1978.

    .. [JB_1995]  Jansen, B., H. and Rit V., G., *Electroencephalogram and
        visual evoked potential generation in a mathematical model of
        coupled cortical columns*, Biological Cybernetics (73) 357:366, 1995.

    .. [JB_1993] Jansen, B., Zouridakis, G. and Brandt, M., *A
        neurophysiologically-based mathematical model of flash visual evoked
        potentials*

    .. [M_2007] Moran

    .. [S_2010] Spiegler

    .. [A_2012] Auburn

    .. figure :: img/ZetterbergJansen_01_mode_0_pplane.svg
        :alt: Jansen and Rit phase plane

    """

    _ui_name = "Zetterberg-Jansen"
    ui_configurable_parameters = ['He', 'Hi', 'ke', 'ki', 'e0', 'rho_2', 'rho_1', 'gamma_1',
                                  'gamma_2', 'gamma_3', 'gamma_4', 'gamma_5', 'P', 'U', 'Q']

    #Define traited attributes for this model, these represent possible kwargs.
    He = numpy.array([3.25])

    Hi = numpy.array([22.0])

    ke = numpy.array([0.1])

    ki = numpy.array([0.05])


    e0 = numpy.array([0.0025])


    rho_2 = numpy.array([6.0])

    rho_1 = numpy.array([0.56])

    gamma_1 = numpy.array([135.0])

    gamma_2 = numpy.array([108.])

    gamma_3 = numpy.array([33.75])

    gamma_4 = numpy.array([33.75])


    gamma_5 = numpy.array([15])

    gamma_1T = numpy.array([1.0])

    gamma_2T = numpy.array([1.0])

    gamma_3T = numpy.array([1.0])

    P = numpy.array([0.12])

    U = numpy.array([0.12])

    Q = numpy.array([0.12])

    #Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = {"v1": numpy.array([-100.0, 100.0]),
             "y1": numpy.array([-500.0, 500.0]),
             "v2": numpy.array([-100.0, 50.0]),
             "y2": numpy.array([-100.0, 6.0]),
             "v3": numpy.array([-100.0, 6.0]),
             "y3": numpy.array([-100.0, 6.0]),
             "v4": numpy.array([-100.0, 20.0]),
             "y4": numpy.array([-100.0, 20.0]),
             "v5": numpy.array([-100.0, 20.0]),
             "y5": numpy.array([-500.0, 500.0]),
             "v6": numpy.array([-100.0, 20.0]),
             "v7": numpy.array([-100.0, 20.0]),}
    # The values for each state-variable should be set to encompass
    # the expected dynamic range of that state-variable for the current
    # parameters, it is used as a mechanism for bounding random inital
    # conditions when the simulation isn't started from an explicit history,
    # it is also provides the default range of phase-plane plots.

    @unique
    class Variables(Enum):
        V1 = "v1"
        Y1 = "y1"
        V2 = "v2"
        Y2 = "y2"
        V3 = "v3"
        Y3 = "y3"
        V4 = "v4"
        Y4 = "y4"
        V5 = "v5"
        Y5 = "y5"
        V6 = "v6"
        V7 = "v7"

        def __get__(self, obj, type):
            return self.value



    state_variables = 'v1 y1 v2 y2 v3 y3 v4 y4 v5 y5 v6 v7'.split()
    _nvar = 12
    cvar = numpy.array([10], dtype=numpy.int32)
    Heke = None  # self.He * self.ke
    Hiki = None  # self.Hi * self.ki
    ke_2 = None  # 2 * self.ke
    ki_2 = None  # 2 * self.ki
    keke = None  # self.ke **2
    kiki = None  # self.ki **2

    def __init__(self, variables_of_interest=None, *args, **kwargs):
        if variables_of_interest is None:
            variables_of_interest = [self.Variables.V6, self.Variables.V7, self.Variables.V2,
                                     self.Variables.V3, self.Variables.V4, self.Variables.V5]
        super(ZetterbergJansen, self).__init__(variables_of_interest=variables_of_interest, *args, **kwargs)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        magic_exp_number = 709

        v1 = state_variables[0, :]
        y1 = state_variables[1, :]
        v2 = state_variables[2, :]
        y2 = state_variables[3, :]
        v3 = state_variables[4, :]
        y3 = state_variables[5, :]
        v4 = state_variables[6, :]
        y4 = state_variables[7, :]
        v5 = state_variables[8, :]
        y5 = state_variables[9, :]
        v6 = state_variables[10, :]
        v7 = state_variables[11, :]

        derivative = numpy.empty_like(state_variables)
        # NOTE: long_range_coupling term: coupling variable is v6 . EQUATIONS
        #       ASSUME linear coupling is used. 'coupled_input' represents a rate. It
        #       is very likely that coeffs gamma_xT should be independent for each of the
        #       terms considered as extrinsic input (P, Q, U) (long range coupling) (local coupling)
        #       and noise.

        coupled_input =  self.sigma_fun(coupling[0, :] + local_coupling * v6)

        # exc input to the excitatory interneurons
        derivative[0] = y1
        derivative[1] = self.Heke * (self.gamma_1 * self.sigma_fun(v2 - v3) + self.gamma_1T * (self.U + coupled_input )) - self.ke_2 * y1 - self.keke * v1
        # exc input to the pyramidal cells
        derivative[2] = y2
        derivative[3] = self.Heke * (self.gamma_2 * self.sigma_fun(v1)      + self.gamma_2T * (self.P + coupled_input )) - self.ke_2 * y2 - self.keke * v2
        # inh input to the pyramidal cells
        derivative[4] = y3
        derivative[5] = self.Hiki * (self.gamma_4 * self.sigma_fun(v4 - v5)) - self.ki_2 * y3 - self.kiki * v3
        derivative[6] = y4
        # exc input to the inhibitory interneurons
        derivative[7] = self.Heke * (self.gamma_3 * self.sigma_fun(v2 - v3) + self.gamma_3T * (self.Q + coupled_input)) - self.ke_2 * y4 - self.keke * v4
        derivative[8] = y5
        # inh input to the inhibitory interneurons
        derivative[9] = self.Hiki * (self.gamma_5 * self.sigma_fun(v4 - v5)) - self.ki_2 * y5 - self.keke * v5
        # aux variables (the sum gathering the postsynaptic inh & exc potentials)
        # pyramidal cells
        derivative[10] = y2 - y3
        # inhibitory cells
        derivative[11] = y4 - y5
        return derivative

    def sigma_fun(self, sv):
        """
        Neuronal activation function. This sigmoidal function
        increases from 0 to Q_max as "sv" increases.
        sv represents a membrane potential state variable (V).

        """
        #HACKERY: Hackery for exponential s that blow up.
        # Set to inf, so the result will be effectively zero.
        magic_exp_number = 709
        temp = self.rho_1 * (self.rho_2 - sv)
        temp = numpy.where(temp > magic_exp_number, numpy.inf, temp)
        sigma_v = (2* self.e0) / (1 + numpy.exp(temp))
        return sigma_v

    def update_derived_parameters(self):
        self.Heke = self.He * self.ke
        self.Hiki = self.Hi * self.ki
        self.ke_2 = 2 * self.ke
        self.ki_2 = 2 * self.ki
        self.keke = self.ke**2
        self.kiki = self.ki**2
