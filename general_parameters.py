import logging
import datetime
import os
import torch
from additional_utils import get_unequaly_distribution_points
from B_Splines import B_Splines
LOG_LEVEL = logging.DEBUG
LOG_LEVEL = logging.INFO

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

OUT_DATA_NAME = 'out_data'
OUT_DATA_FOLDER = f'./{OUT_DATA_NAME}/' + TIMESTAMP

#create folder for images
if not os.path.exists(OUT_DATA_FOLDER):
    os.makedirs(OUT_DATA_FOLDER)

# set up logging
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
#with loglevel
formatter = logging.Formatter('%(levelname)s -- %(message)s')
file_handler = logging.FileHandler(f'{OUT_DATA_FOLDER}/text_log_{TIMESTAMP}.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logging.basicConfig(format='%(levelname)s -- %(message)s')

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
    def __init__(self, args):
        self.length = args.length
        self.total_time = args.total_time
        self.n_points_x = args.n_points_x
        self.n_points_t = args.n_points_t
        self.n_points_init = args.n_points_init
        self.weight_interior = args.weight_interior
        self.weight_initial = args.weight_initial
        self.weight_boundary = args.weight_boundary
        self.layers = args.layers
        self.neurons_per_layer = args.neurons_per_layer
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.eps_interior = args.eps_interior
        self.spline_degree = args.spline_degree
        self.save = args.save
        self.one_dimension = args.one_dimension
        self.uneven_distribution = args.uneven_distribution
        self.device = args.device
        self.splines = args.splines
        self.pinn_is_solution = args.pinn_is_solution
        self.pinn_learns_coeff = args.pinn_learns_coeff
        self.optimize_test_function = args.optimize_test_function
        self.epsilon_list = args.epsilon_list
        self.test_function = None

    def precalculate(self):
        if self.pinn_is_solution or self.splines or self.pinn_learns_coeff:
            self.knot_vector = torch.linspace(0, 1, self.n_points_x)
            self.knot_vector = torch.cat((torch.zeros(self.spline_degree-1), self.knot_vector, torch.ones(self.spline_degree-1)))
            self.n_coeff = len(self.knot_vector) - self.spline_degree - 1
        if general_parameters.optimize_test_function:
            self.test_function = B_Splines(
                    general_parameters.knot_vector,
                    general_parameters.spline_degree,
                    dims=1 if general_parameters.one_dimension else 2
                )

            
general_parameters = GeneralParameters()
