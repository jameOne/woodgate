"""
woodgate_settings_test.py - The woodgate_settings_test.py module
contains unit tests related to the woodgate_setting.py module.
"""
import uuid
import unittest
from .woodgate_settings import Model


class TestWoodgateSettingsDefaults(unittest.TestCase):
    """
    FileSystemConfigurationDefaultsTest - This class encapsulates
    all logic related to unit testing the
    FileSystemConfiguration class using default file
    system configuration.
    """

    def test_models(self) -> None:
        """

        :return:
        :rtype:
        """
        test_uuid = str(uuid.uuid4())
        model = Model("test", test_uuid)

        self.assertTrue(model.model_uuid, test_uuid)


if __name__ == '__main__':
    unittest.main()
