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

general_parameters = GeneralParameters()