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
Framework methods for the LookUpTables datatype.

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>
"""

import tvb.datatypes.lookup_tables_data as lookup_tables_data


class LookUpTableFramework(lookup_tables_data.LookUpTableData):
    """ 
    This class exists to add framework methods and attributes to LookUpTables
    """
    
    __tablename__ = None
    
    pass


class PsiTableFramework(lookup_tables_data.PsiTableData, LookUpTableFramework):
    """ 
    This class exists to add framework methods and attributes to LookUpTables
    """
    
    __tablename__ = None
    

    pass


class NerfTableFramework(lookup_tables_data.NerfTableData, LookUpTableFramework):
    """ 
    This class exists to add framework methods and attributes to LookUpTables
    """
    
    __tablename__ = None
    

    pass