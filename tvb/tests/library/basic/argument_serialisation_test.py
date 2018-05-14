import pytest
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.basic.arguments_serialisation import parse_slice, \
    preprocess_space_parameters, preprocess_time_parameters
from tvb.basic.traits import exceptions

class TestArguments(BaseTestCase):

    def test_parse_slice(self):
        assert parse_slice("::1, :") == (slice(None, None, 1), slice(None, None, None))
        assert parse_slice("2") == 2
        assert parse_slice("[2]") == 2
        assert parse_slice("[]") == slice(None)
        try:
            parse_slice(":::::")
        except ValueError:
            pass

    def test_preprocess_space_parameters(self):
        assert preprocess_space_parameters(1,2,3,10,10,10) == (1,2,6)
        assert preprocess_space_parameters(1.5,2.5,3.5,10,10,10) == (1,2,6)
        try:
            preprocess_space_parameters(20,1,1,10,10,10)
        except exceptions.ValidationException:
            pass

    def test_preprocess_time_parameters(self):
        assert preprocess_time_parameters(10, 20, 50) == (10, 20, 10)
        assert preprocess_time_parameters(10.25, 20.5, 50) == (10, 20, 10)
        try:
            preprocess_time_parameters(10, 20, 5)
        except exceptions.ValidationException:
            pass
        try:
            preprocess_time_parameters(20,10,30)
        except exceptions.ValidationException:
            pass
