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
Models based on Wong-Wang's work.

"""

from .base import ModelNumbaDfun, LOG, numpy, arrays
from numba import guvectorize, float64
from enum import Enum, unique

@guvectorize([(float64[:],)*11], '(n),(m)' + ',()'*8 + '->(n)', nopython=True)
def _numba_dfun(S, c, a, b, d, g, ts, w, j, io, dx):
    "Gufunc for reduced Wong-Wang model equations."

    if S[0] < 0.0:
        dx[0] = 0.0 - S[0]
    elif S[0] > 1.0:
        dx[0] = 1.0 - S[0]
    else:
        x = w[0]*j[0]*S[0] + io[0] + j[0]*c[0]
        h = (a[0]*x - b[0]) / (1 - numpy.exp(-d[0]*(a[0]*x - b[0])))
        dx[0] = - (S[0] / ts[0]) + (1.0 - S[0]) * h * g[0]


class ReducedWongWang(ModelNumbaDfun):
    r"""
    .. [WW_2006] Kong-Fatt Wong and Xiao-Jing Wang,  *A Recurrent Network
                Mechanism of Time Integration in Perceptual Decisions*.
                Journal of Neuroscience 26(4), 1314-1328, 2006.

    .. [DPA_2013] Deco Gustavo, Ponce Alvarez Adrian, Dante Mantini, Gian Luca
                  Romani, Patric Hagmann and Maurizio Corbetta. *Resting-State
                  Functional Connectivity Emerges from Structurally and
                  Dynamically Shaped Slow Linear Fluctuations*. The Journal of
                  Neuroscience 32(27), 11239-11252, 2013.



    .. automethod:: ReducedWongWang.__init__

    Equations taken from [DPA_2013]_ , page 11242

    .. math::
                 x_k       &=   w\,J_N \, S_k + I_o + J_N \mathbf\Gamma(S_k, S_j, u_{kj}),\\
                 H(x_k)    &=  \dfrac{ax_k - b}{1 - \exp(-d(ax_k -b))},\\
                 \dot{S}_k &= -\dfrac{S_k}{\tau_s} + (1 - S_k) \, H(x_k) \, \gamma

    """
    _ui_name = "Reduced Wong-Wang"
    ui_configurable_parameters = ['a', 'b', 'd', 'gamma', 'tau_s', 'w', 'J_N', 'I_o']

    #Define traited attributes for this model, these represent possible kwargs.
    a = numpy.array([0.270, ])

    b = numpy.array([0.108, ])

    d = numpy.array([154., ])

    gamma = numpy.array([0.641, ])

    tau_s = numpy.array([100., ])

    w = numpy.array([0.6, ])

    J_N = numpy.array([0.2609, ])

    I_o = numpy.array([0.33, ])

    sigma_noise = numpy.array([0.000000001, ])

    state_variable_range = {"S": numpy.array([0.0, 1.0])} # Population firing rate

    @unique
    class Variables(Enum):
        S = "S"

        def __get__(self, obj, type):
            return self.value



    state_variables = ['S']
    _nvar = 1
    cvar = numpy.array([0], dtype=numpy.int32)

    def __init__(self, variables_of_interest=None, *args, **kwargs):
        if variables_of_interest is None:
            variables_of_interest = [self.Variables.S]
        super(ReducedWongWang, self).__init__(variables_of_interest=variables_of_interest, *args, **kwargs)

    def configure(self):
        """  """
        super(ReducedWongWang, self).configure()
        self.update_derived_parameters()

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        Equations taken from [DPA_2013]_ , page 11242

        .. math::
                 x_k       &=   w\,J_N \, S_k + I_o + J_N \mathbf\Gamma(S_k, S_j, u_{kj}),\\
                 H(x_k)    &=  \dfrac{ax_k - b}{1 - \exp(-d(ax_k -b))},\\
                 \dot{S}_k &= -\dfrac{S_k}{\tau_s} + (1 - S_k) \, H(x_k) \, \gamma

        """
        S   = state_variables[0, :]
        S[S<0] = 0.
        S[S>1] = 1.
        c_0 = coupling[0, :]


        # if applicable
        lc_0 = local_coupling * S

        x  = self.w * self.J_N * S + self.I_o + self.J_N * c_0 + self.J_N * lc_0
        H = (self.a * x - self.b) / (1 - numpy.exp(-self.d * (self.a * x - self.b)))
        dS = - (S / self.tau_s) + (1 - S) * H * self.gamma

        derivative = numpy.array([dS])
        return derivative

    def dfun(self, x, c, local_coupling=0.0):
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T + local_coupling * x[0]
        deriv = _numba_dfun(x_, c_, self.a, self.b, self.d, self.gamma,
                        self.tau_s, self.w, self.J_N, self.I_o)
        return deriv.T[..., numpy.newaxis]
