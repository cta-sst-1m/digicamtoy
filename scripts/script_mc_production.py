import logging
import os
import sys
from optparse import OptionParser
from yaml import load

from digicamtoy.utils import logger

if __name__ == '__main__':

    parser = OptionParser()

    # Job configuration (the only mandatory option)
    parser.add_option("-y", "--yaml_config", dest="yaml_config",
                      help="full path of the yaml configuration function",
                      default='options/trigger_nsb.yaml')

    # Other options allows to overwrite the yaml_config interactively

    # Output level
    parser.add_option("-v", "--verbose",
                      action="store_false", dest="verbose", default=True,
                      help="move to debug")

    # Steering of the passes
    parser.add_option("-c", "--create_dataset", dest="create_dataset", action="store_true",
                      help="create the MC dataset")

    parser.add_option("-d", "--display_results", dest="display_results", action="store_true",
                      help="display the result of the MC production")

    # Logfile basename
    parser.add_option("-l", "--log_file_basename", dest="log_file_basename",
                      help="string to appear in the log file name")

    (options, args) = parser.parse_args()

    # Load the YAML configuration
    options_yaml = {}
    with open(options.yaml_config) as f:
        options_yaml.update(load(f))


    # Update with interactive options
    for key,val in options_yaml.items():
        if not (key in options.__dict__.keys()): # and (options.__dict__[key])):
            options.__dict__[key]=val
        else:
            options_yaml[key]=options.__dict__[key]

    __name__ = options.production_module
    # load the analysis module
    logger.initialise_logger(options)
    print('--------------------------', options.production_module)
    production_module = __import__('production.%s' % options.production_module,\
                                     locals=None, \
                                     globals=None, \
                                     fromlist=[None], \
                                     level=0)

    # Some logging
    log = logging.getLogger(sys.modules['__main__'].__name__)
    log.info('\t\t-|> Will run %s with the following configuration:' % options.production_module)
    for key, val in options_yaml.items():
        log.info('\t\t |--|> %s : \t %s' % (key, val))
    log.info('-|')

    if options.create_dataset:
        # Call the creation function

        if not os.path.exists(options.output_directory + options.file_basename):
            log.info('\t\t-|> Create the Monte Carlo dataset')
            production_module.create_dataset(options)

        else:

            log.error('File %s already exists' % (options.output_directory + options.file_basename))

    if options.display_results:

        import matplotlib.pyplot as plt
        plt.ion()
        # Call the display function
        log.info('\t\t-|> Display the Monte Carlo dataset')
        production_module.display(options)
