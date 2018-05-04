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
Larter-Breakspear model based on the Morris-Lecar equations.

"""

from .base import Model, LOG, numpy, arrays
from enum import Enum, unique

class LarterBreakspear(Model):
    r"""
    A modified Morris-Lecar model that includes a third equation which simulates
    the effect of a population of inhibitory interneurons synapsing on
    the pyramidal cells.

    .. [Larteretal_1999] Larter et.al. *A coupled ordinary differential equation
        lattice model for the simulation of epileptic seizures.* Chaos. 9(3):
        795, 1999.

    .. [Breaksetal_2003_a] Breakspear, M.; Terry, J. R. & Friston, K. J.  *Modulation of excitatory
        synaptic coupling facilitates synchronization and complex dynamics in an
        onlinear model of neuronal dynamics*. Neurocomputing 52–54 (2003).151–158

    .. [Breaksetal_2003_b] M. J. Breakspear et.al. *Modulation of excitatory
        synaptic coupling facilitates synchronization and complex dynamics in a
        biophysical model of neuronal dynamics.* Network: Computation in Neural
        Systems 14: 703-732, 2003.

    .. [Honeyetal_2007] Honey, C.; Kötter, R.; Breakspear, M. & Sporns, O. * Network structure of
        cerebral cortex shapes functional connectivity on multiple time scales*. (2007)
        PNAS, 104, 10240

    .. [Honeyetal_2009] Honey, C. J.; Sporns, O.; Cammoun, L.; Gigandet, X.; Thiran, J. P.; Meuli,
        R. & Hagmann, P. *Predicting human resting-state functional connectivity
        from structural connectivity.* (2009), PNAS, 106, 2035-2040

    .. [Alstottetal_2009] Alstott, J.; Breakspear, M.; Hagmann, P.; Cammoun, L. & Sporns, O.
        *Modeling the impact of lesions in the human brain*. (2009)),  PLoS Comput Biol, 5, e1000408

    Equations and default parameters are taken from [Breaksetal_2003_b]_.
    All equations and parameters are non-dimensional and normalized.
    For values of d_v  < 0.55, the dynamics of a single column settles onto a
    solitary fixed point attractor.


    Parameters used for simulations in [Breaksetal_2003_a]_ Table 1. Page 153.
    Two nodes were coupled. C=0.1

    +---------------------------+
    |          Table 1          |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | I            |      0.3   |
    +--------------+------------+
    | a_ee         |      0.4   |
    +--------------+------------+
    | a_ei         |      0.1   |
    +--------------+------------+
    | a_ie         |      1.0   |
    +--------------+------------+
    | a_ne         |      1.0   |
    +--------------+------------+
    | a_ni         |      0.4   |
    +--------------+------------+
    | r_NMDA       |      0.2   |
    +--------------+------------+
    | delta        |      0.001 |
    +--------------+------------+
    |   Breakspear et al. 2003  |
    +---------------------------+


    +---------------------------+
    |          Table 2          |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | gK           |      2.0   |
    +--------------+------------+
    | gL           |      0.5   |
    +--------------+------------+
    | gNa          |      6.7   |
    +--------------+------------+
    | gCa          |      1.0   |
    +--------------+------------+
    | a_ne         |      1.0   |
    +--------------+------------+
    | a_ni         |      0.4   |
    +--------------+------------+
    | a_ee         |      0.36  |
    +--------------+------------+
    | a_ei         |      2.0   |
    +--------------+------------+
    | a_ie         |      2.0   |
    +--------------+------------+
    | VK           |     -0.7   |
    +--------------+------------+
    | VL           |     -0.5   |
    +--------------+------------+
    | VNa          |      0.53  |
    +--------------+------------+
    | VCa          |      1.0   |
    +--------------+------------+
    | phi          |      0.7   |
    +--------------+------------+
    | b            |      0.1   |
    +--------------+------------+
    | I            |      0.3   |
    +--------------+------------+
    | r_NMDA       |      0.25  |
    +--------------+------------+
    | C            |      0.1   |
    +--------------+------------+
    | TCa          |     -0.01  |
    +--------------+------------+
    | d_Ca         |      0.15  |
    +--------------+------------+
    | TK           |      0.0   |
    +--------------+------------+
    | d_K          |      0.3   |
    +--------------+------------+
    | VT           |      0.0   |
    +--------------+------------+
    | ZT           |      0.0   |
    +--------------+------------+
    | TNa          |      0.3   |
    +--------------+------------+
    | d_Na         |      0.15  |
    +--------------+------------+
    | d_V          |      0.65  |
    +--------------+------------+
    | d_Z          |      d_V   |
    +--------------+------------+
    | QV_max       |      1.0   |
    +--------------+------------+
    | QZ_max       |      1.0   |
    +--------------+------------+
    |   Alstott et al. 2009     |
    +---------------------------+


    NOTES about parameters

    :math:`\delta_V` : for :math:`\delta_V` < 0.55, in an uncoupled network,
    the system exhibits fixed point dynamics; for 0.55 < :math:`\delta_V` < 0.59,
    limit cycle attractors; and for :math:`\delta_V` > 0.59 chaotic attractors
    (eg, d_V=0.6,aee=0.5,aie=0.5, gNa=0, Iext=0.165)

    :math:`\delta_Z`
    this parameter might be spatialized: ones(N,1).*0.65 + modn*(rand(N,1)-0.5);

    :math:`C`
    The long-range coupling :math:`\delta_C` is ‘weak’ in the sense that
    the model is well behaved for parameter values for which C < a_ee and C << a_ie.



    .. figure :: img/LarterBreakspear_01_mode_0_pplane.svg
            :alt: Larter-Breaskpear phase plane (V, W)

            The (:math:`V`, :math:`W`) phase-plane for the Larter-Breakspear model.

    .. automethod:: LarterBreakspear.__init__

    Dynamic equations:

    .. math::
            \dot{V}_k & = - (g_{Ca} + (1 - C) \, r_{NMDA} \, a_{ee} \, Q_V + C \, r_{NMDA} \, a_{ee} \, \langle Q_V\rangle^{k}) \, m_{Ca} \, (V - VCa) \\
                           & \,\,- g_K \, W \, (V - VK) -  g_L \, (V - VL) \\
                           & \,\,- (g_{Na} \, m_{Na} + (1 - C) \, a_{ee} \, Q_V + C \, a_{ee} \, \langle Q_V\rangle^{k}) \,(V - VNa) \\
                           & \,\,- a_{ie} \, Z \, Q_Z + a_{ne} \, I, \\
                           & \\
            \dot{W}_k & = \phi \, \dfrac{m_K - W}{\tau_{K}},\\
                           & \nonumber\\
            \dot{Z}_k &= b (a_{ni}\, I + a_{ei}\,V\,Q_V),\\
            Q_{V}   &= Q_{V_{max}} \, (1 + \tanh\left(\dfrac{V_{k} - VT}{\delta_{V}}\right)),\\
            Q_{Z}   &= Q_{Z_{max}} \, (1 + \tanh\left(\dfrac{Z_{k} - ZT}{\delta_{Z}}\right)),

        See Equations (7), (3), (6) and (2) respectively in [Breaksetal_2003_a]_.
        Pag: 705-706

    """

    _ui_name = "Larter-Breakspear"
    ui_configurable_parameters = ['gCa', 'gK', 'gL', 'phi', 'gNa', 'TK', 'TCa',
                                  'TNa', 'VCa', 'VK', 'VL', 'VNa', 'd_K', 'tau_K',
                                  'd_Na', 'd_Ca', 'aei', 'aie', 'b', 'C', 'ane',
                                  'ani', 'aee', 'Iext', 'rNMDA', 'VT', 'd_V', 'ZT',
                                  'd_Z', 'QV_max', 'QZ_max', 't_scale']

    #Define traited attributes for this model, these represent possible kwargs.
    gCa = numpy.array([1.1])

    gK = numpy.array([2.0])

    gL = numpy.array([0.5])

    phi = numpy.array([0.7])

    gNa = numpy.array([6.7])

    TK = numpy.array([0.0])

    TCa = numpy.array([-0.01])

    TNa = numpy.array([0.3])

    VCa = numpy.array([1.0])

    VK = numpy.array([-0.7])

    VL = numpy.array([-0.5])

    VNa = numpy.array([0.53])

    d_K = numpy.array([0.3])

    tau_K = numpy.array([1.0])

    d_Na = numpy.array([0.15])

    d_Ca =  numpy.array([0.15])

    aei = numpy.array([2.0])

    aie = numpy.array([2.0])

    b = numpy.array([0.1])

    C = numpy.array([0.1])

    ane = numpy.array([1.0])

    ani = numpy.array([0.4])

    aee = numpy.array([0.4])

    Iext = numpy.array([0.3])

    rNMDA = numpy.array([0.25])

    VT = numpy.array([0.0])

    d_V = numpy.array([0.65])

    ZT = numpy.array([0.0])

    d_Z = numpy.array([0.7])

    # NOTE: the values were not in the article.
    QV_max = numpy.array([1.0])

    QZ_max = numpy.array([1.0])


    t_scale = numpy.array([1.0])

    @unique
    class Variables(Enum):
        V = "V"
        W = "W"
        Z = "Z"

        def __get__(self, obj, type):
            return self.value


    #Informational attribute, used for phase-plane and initial()
    state_variable_range = {"V": numpy.array([-1.5, 1.5]),
                            "W": numpy.array([-1.5, 1.5]),
                            "Z": numpy.array([-1.5, 1.5])}

    # The values for each state-variable should be set to encompass
    #     the expected dynamic range of that state-variable for the current
    #     parameters, it is used as a mechanism for bounding random inital
    #     conditions when the simulation isn't started from an explicit
    #     history, it is also provides the default range of phase-plane plots

    state_variables = 'V W Z'.split()
    _state_variables = ["V", "W", "Z"]
    _nvar = 3
    cvar = numpy.array([0], dtype=numpy.int32)

    def __init__(self, variables_of_interest=None, *args, **kwargs):
        if variables_of_interest is None:
            variables_of_interest = [self.Variables.V]
        super(LarterBreakspear, self).__init__(variables_of_interest=variables_of_interest, *args, **kwargs)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        Dynamic equations:

        .. math::
            \dot{V}_k & = - (g_{Ca} + (1 - C) \, r_{NMDA} \, a_{ee} \, Q_V + C \, r_{NMDA} \, a_{ee} \, \langle Q_V\rangle^{k}) \, m_{Ca} \, (V - VCa) \\
                           & \,\,- g_K \, W \, (V - VK) -  g_L \, (V - VL) \\
                           & \,\,- (g_{Na} \, m_{Na} + (1 - C) \, a_{ee} \, Q_V + C \, a_{ee} \, \langle Q_V\rangle^{k}) \,(V - VNa) \\
                           & \,\,- a_{ie} \, Z \, Q_Z + a_{ne} \, I, \\
                           & \\
            \dot{W}_k & = \phi \, \dfrac{m_K - W}{\tau_{K}},\\
                           & \nonumber\\
            \dot{Z}_k &= b (a_{ni}\, I + a_{ei}\,V\,Q_V),\\
            Q_{V}   &= Q_{V_{max}} \, (1 + \tanh\left(\dfrac{V_{k} - VT}{\delta_{V}}\right)),\\
            Q_{Z}   &= Q_{Z_{max}} \, (1 + \tanh\left(\dfrac{Z_{k} - ZT}{\delta_{Z}}\right)),

        """
        V, W, Z = state_variables
        derivative = numpy.empty_like(state_variables)
        c_0   = coupling[0, :]
        # relationship between membrane voltage and channel conductance
        m_Ca = 0.5 * (1 + numpy.tanh((V - self.TCa) / self.d_Ca))
        m_Na = 0.5 * (1 + numpy.tanh((V - self.TNa) / self.d_Na))
        m_K  = 0.5 * (1 + numpy.tanh((V - self.TK )  / self.d_K))
        # voltage to firing rate
        QV    = 0.5 * self.QV_max * (1 + numpy.tanh((V - self.VT) / self.d_V))
        QZ    = 0.5 * self.QZ_max * (1 + numpy.tanh((Z - self.ZT) / self.d_Z))
        lc_0  = local_coupling * QV
        derivative[0] = self.t_scale * (- (self.gCa + (1.0 - self.C) * (self.rNMDA * self.aee) * (QV + lc_0)+ self.C * self.rNMDA * self.aee * c_0) * m_Ca * (V - self.VCa)
                         - self.gK * W * (V - self.VK)
                         - self.gL * (V - self.VL)
                         - (self.gNa * m_Na + (1.0 - self.C) * self.aee * (QV  + lc_0) + self.C * self.aee * c_0) * (V - self.VNa)
                         - self.aie * Z * QZ
                         + self.ane * self.Iext)
        derivative[1] = self.t_scale * self.phi * (m_K - W) / self.tau_K
        derivative[2] = self.t_scale * self.b * (self.ani * self.Iext + self.aei * V * QV)
        return derivative
