$epochs = 440
$layers = 5
$neurons_per_layer = 20
$learning_rate = 0.01
$spline_degree = 4
$weight_interior = 1.0
$weight_boundary = 1.0
$weight_initial = 0.0


############################################################################################################

python main_solution.py `
    --n_points_x 10 `
    --epochs $epochs `
    --eps_interior 1.0 `
    --weight_interior $weight_interior `
    --weight_boundary $weight_boundary `
    --weight_initial $weight_initial `
    --layers $layers  `
    --neurons_per_layer $neurons_per_layer  `
    --learning_rate $learning_rate `
    --spline_degree $spline_degree `
    --one_dimension `
    --optimize_test_function `
    --pinn_is_solution `
    --save

######################################################

python main_solution.py `
    --n_points_x 100 `
    --epochs $epochs `
    --eps_interior 1.0 `
    --weight_interior $weight_interior  `
    --weight_boundary $weight_boundary `
    --weight_initial $weight_initial `
    --layers  $layers  `
    --neurons_per_layer $neurons_per_layer  `
    --learning_rate $learning_rate `
    --spline_degree $spline_degree `
    --one_dimension `
    --optimize_test_function `
    --pinn_is_solution `
    --save

######################################################

python main_solution.py `
    --n_points_x 1000 `
    --epochs $epochs `
    --eps_interior 1.0 `
    --weight_interior $weight_interior  `
    --weight_boundary $weight_boundary `
    --weight_initial $weight_initial `
    --layers  $layers  `
    --neurons_per_layer $neurons_per_layer  `
    --learning_rate $learning_rate `
    --spline_degree $spline_degree `
    --one_dimension `
    --optimize_test_function `
    --pinn_is_solution `
    --save

############################################################################################################

python main_solution.py `
    --n_points_x 10 `
    --epochs $epochs `
    --eps_interior 0.1 `
    --weight_interior $weight_interior  `
    --weight_boundary $weight_boundary `
    --weight_initial $weight_initial `
    --layers  $layers  `
    --neurons_per_layer $neurons_per_layer  `
    --learning_rate $learning_rate `
    --spline_degree $spline_degree `
    --one_dimension `
    --optimize_test_function `
    --pinn_is_solution `
    --save

######################################################

python main_solution.py `
    --n_points_x 100 `
    --epochs $epochs `
    --eps_interior 0.1 `
    --weight_interior $weight_interior  `
    --weight_boundary $weight_boundary `
    --weight_initial $weight_initial `
    --layers  $layers  `
    --neurons_per_layer $neurons_per_layer  `
    --learning_rate $learning_rate `
    --spline_degree $spline_degree `
    --one_dimension `
    --optimize_test_function `
    --pinn_is_solution `
    --save

######################################################

python main_solution.py `
    --n_points_x 1000 `
    --epochs $epochs `
    --eps_interior 0.1 `
    --weight_interior $weight_interior  `
    --weight_boundary $weight_boundary `
    --weight_initial $weight_initial `
    --layers  $layers  `
    --neurons_per_layer $neurons_per_layer  `
    --learning_rate $learning_rate `
    --spline_degree $spline_degree `
    --one_dimension `
    --optimize_test_function `
    --pinn_is_solution `
    --save

############################################################################################################

python main_solution.py `
    --n_points_x 10 `
    --epochs $epochs `
    --eps_interior 0.01 `
    --weight_interior $weight_interior  `
    --weight_boundary $weight_boundary `
    --weight_initial $weight_initial `
    --layers  $layers  `
    --neurons_per_layer $neurons_per_layer  `
    --learning_rate $learning_rate `
    --spline_degree $spline_degree `
    --one_dimension `
    --optimize_test_function `
    --pinn_is_solution `
    --save

######################################################

python main_solution.py `
    --n_points_x 100 `
    --epochs $epochs `
    --eps_interior 0.01 `
    --weight_interior $weight_interior  `
    --weight_boundary $weight_boundary `
    --weight_initial $weight_initial `
    --layers  $layers  `
    --neurons_per_layer $neurons_per_layer  `
    --learning_rate $learning_rate `
    --spline_degree $spline_degree `
    --one_dimension `
    --optimize_test_function `
    --pinn_is_solution `
    --save

######################################################

python main_solution.py `
    --n_points_x 1000 `
    --epochs $epochs `
    --eps_interior 0.01 `
    --weight_interior $weight_interior  `
    --weight_boundary $weight_boundary `
    --weight_initial $weight_initial `
    --layers  $layers  `
    --neurons_per_layer $neurons_per_layer  `
    --learning_rate $learning_rate `
    --spline_degree $spline_degree `
    --one_dimension `
    --optimize_test_function `
    --pinn_is_solution `
    --save

############################################################################################################