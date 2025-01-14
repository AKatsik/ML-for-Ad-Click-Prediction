""" Secondary functions that are used throughout the ML pipeline. """

import yaml


class Helper:
    """ A class that contains all the helper functions. """

    @staticmethod
    def read_yaml_file(path: str):
        """ Reads a yaml file and returns its content. """
        with open(path, encoding="utf-8") as file:
            constent = yaml.load(file, Loader=yaml.FullLoader)
            file.close()
        return constent