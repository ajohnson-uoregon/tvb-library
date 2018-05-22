from tvb.tests.library.base_testcase import BaseTestCase
from tvb.datatypes.region_mapping import RegionMapping

class TestRegionMapping(BaseTestCase):

    def test_regionmapping(self):
        dt = RegionMapping(load_file="regionMapping_16k_76.txt")
        assert isinstance(dt, RegionMapping)
        assert dt.mapping.shape == (16384,)
