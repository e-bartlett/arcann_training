"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2023/09/04
Last modified: 2023/09/04

Test cases for the machine module.

Classes
-------
TestGetHostName():
    Test cases for the 'get_host_name' function.

TestAssertSameMachine():
    Test cases for the 'assert_same_machine' function.
    
TestGetMachineKeyword():
    Test cases for the 'get_machine_keyword' function.

TestGetMachineFromConfigs():
    Test cases for the 'get_machine_from_configs' function.

"""
# Standard library modules
import unittest
import socket
from unittest.mock import patch

# Local imports
from deepmd_iterative.common.machine import (
    get_host_name,
    assert_same_machine,
    get_machine_keyword,
    get_machine_config_files,
    get_machine_from_configs,
    get_machine_spec_for_step,
)


class TestGetHostName(unittest.TestCase):
    """
    Test cases for the 'get_host_name' function.

    Methods
    -------
    test_fully_qualified_hostname():
        Test fully-qualified hostname case.
    test_fully_qualified_hostname():
        Test partial hostname case.
    test_fully_qualified_hostname():
        Test timeout exception case.
    """

    @patch("socket.gethostname")
    def test_fully_qualified_hostname(self, mock_gethostname):
        """
        Test fully-qualified hostname case.
        """
        mock_gethostname.return_value = "fully.qualified.hostname"
        result = get_host_name()
        self.assertEqual(result, "fully.qualified.hostname")

    @patch("socket.gethostname")
    @patch("socket.gethostbyaddr")
    def test_partial_hostname(self, mock_gethostbyaddr, mock_gethostname):
        """
        Test partial hostname case.
        """
        mock_gethostname.return_value = "partial"
        mock_gethostbyaddr.return_value = ("fully.qualified.partial", [], [])
        result = get_host_name()
        self.assertEqual(result, "fully.qualified.partial")

    @patch("socket.gethostname")
    @patch("socket.gethostbyaddr")
    def test_timeout_exception(self, mock_gethostbyaddr, mock_gethostname):
        """
        Test timeout exception case.
        """
        mock_gethostname.return_value = "partial"
        mock_gethostbyaddr.side_effect = socket.timeout
        result = get_host_name()
        self.assertEqual(result, "partial")


class TestAssertSameMachine(unittest.TestCase):
    """
    Test cases for the 'assert_same_machine' function.

    Methods
    -------
    test_matching_machine_names():
        Test case with matching machine names.
    test_non_matching_machine_names():
        Test case with non-matching machine names.
    """

    def test_matching_machine_names(self):
        """
        Test case with matching machine names.
        """
        expected_machine = "expected_machine"
        machine_config = {"machine": "expected_machine"}

        assert_same_machine(expected_machine, machine_config)

        # No assertions needed since the function should not raise an exception

    def test_non_matching_machine_names(self):
        """
        Test case with non-matching machine names.
        """
        expected_machine = "expected_machine"
        machine_config = {"machine": "different_machine"}

        with self.assertRaises(ValueError):
            assert_same_machine(expected_machine, machine_config)


class TestGetMachineKeyword(unittest.TestCase):
    """
    Test cases for the 'get_machine_keyword' function.

    Methods
    -------
    test_valid_bool_value():
        Test valid boolean value case.
    test_valid_string_value():
        Test valid string value case.
    test_valid_list_value():
        Test valid list value case.
    test_key_not_found():
        Test case where key is not found in any JSON.
    test_invalid_value_type():
        Test case with an invalid value type.
    """

    def test_valid_bool_value(self):
        """
        Test valid boolean value case.
        """
        input_json = {"user_machine_keyword": True}
        previous_json = {}
        default_json = {}

        result = get_machine_keyword(input_json, previous_json, default_json)
        self.assertEqual(result, True)

    def test_valid_string_value(self):
        """
        Test valid string value case.
        """
        input_json = {"user_machine_keyword": "some_keyword"}
        previous_json = {}
        default_json = {}

        result = get_machine_keyword(input_json, previous_json, default_json)
        self.assertEqual(result, "some_keyword")

    def test_valid_list_value(self):
        """
        Test valid list value case.
        """
        input_json = {"user_machine_keyword": ["project", "allocation", "arch_name"]}
        previous_json = {}
        default_json = {}

        result = get_machine_keyword(input_json, previous_json, default_json)
        self.assertEqual(result, ["project", "allocation", "arch_name"])

    def test_key_not_found(self):
        """
        Test case where key is not found in any JSON.
        """
        input_json = {}
        previous_json = {}
        default_json = {}

        with self.assertRaises(KeyError):
            get_machine_keyword(input_json, previous_json, default_json)

    def test_invalid_value_type(self):
        """
        Test case with an invalid value type.
        """
        input_json = {"user_machine_keyword": 123}
        previous_json = {}
        default_json = {}

        with self.assertRaises(TypeError):
            get_machine_keyword(input_json, previous_json, default_json)


class TestGetMachineFromConfigs(unittest.TestCase):
    """
    Test cases for the 'get_machine_from_configs' function.

    Methods
    -------
    test_matching_hostname():
        Test case where a matching hostname is found.
    test_matching_machine_short_name():
        Test case where a matching machine short name is provided.
    test_no_matching_config():
        Test case where no matching configuration is found.
    """

    @patch("deepmd_iterative.common.machine.get_host_name", return_value="my_machine")
    def test_matching_hostname(self, mock_get_host_name):
        """
        Test case where a matching hostname is found.
        """
        machine_configs = [
            {"machine1": {"hostname": "my_machine"}},
            {"machine2": {"hostname": "other_machine"}},
        ]
        result = get_machine_from_configs(machine_configs)
        self.assertEqual(result, "machine1")

    def test_matching_machine_short_name(self):
        """
        Test case where a matching machine short name is provided.
        """
        machine_configs = [
            {"machine1": {"hostname": "my_machine"}},
            {"machine2": {"hostname": "other_machine"}},
        ]
        result = get_machine_from_configs(
            machine_configs, machine_short_name="machine2"
        )
        self.assertEqual(result, "machine2")

    @patch(
        "deepmd_iterative.common.machine.get_host_name", return_value="unknown_machine"
    )
    def test_no_matching_config(self, mock_get_host_name):
        """
        Test case where no matching configuration is found.
        """
        machine_configs = [
            {"machine1": {"hostname": "my_machine"}},
            {"machine2": {"hostname": "other_machine"}},
        ]
        with self.assertRaises(ValueError):
            get_machine_from_configs(machine_configs)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
