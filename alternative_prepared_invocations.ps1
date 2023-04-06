$epochs = 5000
$layers = 4
$neurons_per_layer = 20
$learning_rate = 0.025
$spline_degree = 3
$weight_interior = 1.0
$weight_boundary = 1.0
$weight_initial = 0.0

$eps_interior_values = @(1.0, 0.1, 0.01)
$n_points_x_values = @(10, 100, 150)

foreach ($eps_interior in $eps_interior_values) {
    foreach ($n_points_x in $n_points_x_values) {
        python main_solution.py `
            --n_points_x $n_points_x `
            --epochs $epochs `
            --eps_interior $eps_interior `
            --weight_interior $weight_interior `
            --weight_boundary $weight_boundary `
            --weight_initial $weight_initial `
            --layers $layers `
            --neurons_per_layer $neurons_per_layer `
            --learning_rate $learning_rate `
            --spline_degree $spline_degree `
            --one_dimension `
            --optimize_test_function `
            --pinn_is_solution `
            --save
    }
}
