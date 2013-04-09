# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#

"""

The Array datatypes. This brings together the scientific and framework 
methods that are associated with the Array datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import tvb.datatypes.arrays_scientific as arrays_scientific
import tvb.datatypes.arrays_framework as arrays_framework
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)


class FloatArray(arrays_scientific.FloatArrayScientific, arrays_framework.FloatArrayFramework):
    """
    This class brings together the scientific and framework methods that are
    associated with the FloatArray datatype.
    
    ::
        
                           FloatArrayData
                                 |
                                / \\
             FloatArrayFramework   FloatArrayScientific
                                \ /
                                 |
                             FloatArray
        
    
    """
    pass


class IntegerArray(arrays_scientific.IntegerArrayScientific,
                 arrays_framework.IntegerArrayFramework):
    """
    This class brings together the scientific and framework methods that are
    associated with the IntegerArray datatype.
    
    ::
        
                          IntegerArrayData
                                 |
                                / \\
           IntegerArrayFramework   IntegerArrayScientific
                                \ /
                                 |
                            IntegerArray
        
    
    """
    pass


class ComplexArray(arrays_scientific.ComplexArrayScientific,
                 arrays_framework.ComplexArrayFramework):
    """
    This class brings together the scientific and framework methods that are
    associated with the ComplexArray datatype.
    
    ::
        
                          ComplexArrayData
                                 |
                                / \\
           ComplexArrayFramework   ComplexArrayScientific
                                \ /
                                 |
                            ComplexArray
        
    
    """
    pass


class BoolArray(arrays_scientific.BoolArrayScientific,
                 arrays_framework.BoolArrayFramework):
    """
    This class brings together the scientific and framework methods that are
    associated with the BoolArray datatype.
    
    ::
        
                           BoolArrayData
                                 |
                                / \\
              BoolArrayFramework   BoolArrayScientific
                                \ /
                                 |
                             BoolArray
        
    
    """
    pass


class StringArray(arrays_scientific.StringArrayScientific,
                 arrays_framework.StringArrayFramework):
    """
    This class brings together the scientific and framework methods that are
    associated with the StringArray datatype.
    
    ::
        
                          StringArrayData
                                 |
                                / \\
            StringArrayFramework   StringArrayScientific
                                \ /
                                 |
                            StringArray
        
    
    """
    pass


class PositionArray(arrays_scientific.PositionArrayScientific,
                 arrays_framework.PositionArrayFramework):
    """
    This class brings together the scientific and framework methods that are
    associated with the PositionArray datatype.
    
    ::
        
                          PositionArrayData
                                 |
                                / \\
          PositionArrayFramework   PositionArrayScientific
                                \ /
                                 |
                            PositionArray
        
    
    """
    pass


class OrientationArray(arrays_scientific.OrientationArrayScientific,
                       arrays_framework.OrientationArrayFramework):
    """
    This class brings together the scientific and framework methods that are
    associated with the OrientationArray datatype.
    
    ::
        
                        OrientationArrayData
                                 |
                                / \\
       OrientationArrayFramework   OrientationArrayScientific
                                \ /
                                 |
                          OrientationArray
        
    
    """
    pass


class IndexArray(arrays_scientific.IndexArrayScientific,
                 arrays_framework.IndexArrayFramework):
    """
    This class brings together the scientific and framework methods that are
    associated with the IndexArray datatype.
    
    ::
        
                           IndexArrayData
                                 |
                                / \\
             IndexArrayFramework   IndexArrayScientific
                                \ /
                                 |
                             IndexArray
        
    
    """
    pass


class MappedArray(arrays_scientific.MappedArrayScientific,
                  arrays_framework.MappedArrayFramework):
    """
    This class brings together the scientific and framework methods that are
    associated with the MappedArray datatype.
    
    ::
        
                           MappedArrayData
                                 |
                                / \\
             MappedArrayFramework   MappedArrayScientific
                                \ /
                                 |
                             MappedArray
        
    """
    pass


if __name__ == '__main__':
    # Do some stuff that tests or makes use of this module...
    LOG.info("Testing %s module..." % __file__)
    
    # Check that all default Array datatypes initializes without error.
    FLOAT_ARRAY = FloatArray()
    INTEGER_ARRAY = IntegerArray()
    COMPLEX_ARRAY = ComplexArray()
    BOOL_ARRAY = BoolArray()
    STRING_ARRAY = StringArray()
    POSITION_ARRAY = PositionArray()
    ORIENTATION_ARRAY = OrientationArray()
    INDEX_ARRAY = IndexArray()
    MAPPED_ARRAY = MappedArray()
    
    LOG.info("Default Array datatypes initialized without error...")
