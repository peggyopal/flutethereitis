import sys, os
module_path = os.path.dirname(os.path.abspath("src/models/process_data.py"))
sys.path.insert(0, module_path + '/../../')

from src.models import process_data

import pytest
import unittest


class TestLookUpLabel:

    def test__lookup_label_by_index_flute_is_flute(self):
        assert process_data._lookup_label_by_index(196) == 'Flute'

    def test__lookup_label_by_index_flute_is_not_didgeridoo(self):
        assert process_data._lookup_label_by_index(196) != 'Didgeridoo'


# class Test
