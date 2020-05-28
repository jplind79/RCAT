"""
Config reader module
--------------------
Creates and return a dictionary built from a config file.

Created: Autumn 2016
Authors: David Lindstedt & Petter Lind
"""


def _check_vals(item):
    """
    Check if item is evaluatable and returns the value of it.
    Example 'False' should return False
            'ABC' should return 'ABC'
    PARAMETERS: String (the value from a key/value pair)
    RETURNS   : type of evaluated string
    """

    import ast

    try:
        val = ast.literal_eval(item)
    except:
        val = item
    return val


def _get_items(lst):
    """
    Create a dictionary based on input list of items (tuple)
    from config (ini) file.
    Returns the dictionary
    INPUT: a list of tuples (key/value)
    RETURNS: Dictionary (nested)
    """

    d = {}
    # For items that needs to be evaluated (lists, booleans etc.)
    # try ast.literal_eval, else (strings) just add them to dict.
    [d.update({item[0]:_check_vals(item[1])}) for item in lst]
    return d


def get_config_dict(ini_file):
    """
    Create a dictionary from then input config file.
    PARAMETERS: config (.ini) file
    RETURNS   : Dictionary
    """
    import configparser
    # create a configparser object and read input file.
    config = configparser.ConfigParser()
    config.read(ini_file)

    # Create a dict of sections and items from an .ini file.
    config_dict = {k: _get_items(config.items(k)) for k in config.sections()}
    return config_dict
