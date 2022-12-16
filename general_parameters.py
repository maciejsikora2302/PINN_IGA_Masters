# LENGTH = 1.
# TOTAL_TIME = 1.
# N_POINTS_X = 150
# N_POINTS_T = 150
# N_POINTS_INIT = 300
# WEIGHT_INTERIOR = 0.5
# WEIGHT_INITIAL = 150.0
# WEIGHT_BOUNDARY = 1.0
# LAYERS = 2
# NEURONS_PER_LAYER = 60
# EPOCHS = 50_000
# LEARNING_RATE = 0.0025
import logging
import datetime
import os
import torch
LOG_LEVEL = logging.INFO

IMGS_FOLDER = './imgs/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#create folder for images
if not os.path.exists(IMGS_FOLDER):
    os.makedirs(IMGS_FOLDER)

# set up logging
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
#with loglevel
formatter = logging.Formatter('%(levelname)s -- %(message)s')
file_handler = logging.FileHandler(f'logs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logging.basicConfig(level=LOG_LEVEL, format='%(levelname)s -- %(message)s')

class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BLACK = '\033[98m'
    RESET = '\033[0m'


class GeneralParameters:
    def __init__(self, \
            length = None, \
            total_time = None, \
            n_points_x = None, \
            n_points_t = None, \
            n_points_init = None, \
            weight_interior = None, \
            weight_initial = None, \
            weight_boundary = None, \
            layers = None, \
            neurons_per_layer = None, \
            epochs = None, \
            learning_rate = None, \
            eps_interior = None, \
            device = None):
        
        self.length = 1. if length is None else length
        self.total_time = 1. if total_time is None else total_time
        self.n_points_x = 150 if n_points_x is None else n_points_x
        self.n_points_t = 150 if n_points_t is None else n_points_t
        self.n_points_init = 300 if n_points_init is None else n_points_init
        self.weight_interior = 0.5 if weight_interior is None else weight_interior
        self.weight_initial = 150.0 if weight_initial is None else weight_initial
        self.weight_boundary = 1.0 if weight_boundary is None else weight_boundary
        self.layers = 2 if layers is None else layers
        self.neurons_per_layer = 60 if neurons_per_layer is None else neurons_per_layer
        self.epochs = 50_000 if epochs is None else epochs
        self.learning_rate = 0.0025 if learning_rate is None else learning_rate
        self.eps_interior = 1e-3 if eps_interior is None else eps_interior
        self.knot_vector_length = int(5 / self.eps_interior)
        self.test_function_weight_x = torch.ones(self.knot_vector_length)
        self.test_function_weight_t = torch.ones(self.knot_vector_length)

general_parameters = GeneralParameters()
