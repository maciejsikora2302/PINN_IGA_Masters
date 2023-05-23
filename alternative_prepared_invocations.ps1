$epochs = 150000
$epochs = 40000
# $epochs = 2000
# $epochs = 100
$layers = 4
$neurons_per_layer = 20
$learning_rate = 0.00125
$spline_degree = 3
$weight_interior = 1.0
$weight_boundary = 1.0
$weight_initial = 0.0

$eps_interior_values = @(0.1, 0.01, 0.001)
$n_points_x_values = @(10, 20, 30)

Write-output "Adam learns coeffs of bsplines"
foreach ($eps_interior in $eps_interior_values) {
    foreach ($n_points_x in $n_points_x_values) {
        ### print invocations to console as strings in one line, so that they can be copied and pasted, add escaped backticks ###
        # "python main_solution.py ```
        #     --n_points_x $n_points_x ```
        #     --epochs $epochs ```
        #     --eps_interior $eps_interior ```
        #     --weight_interior $weight_interior ```
        #     --weight_boundary $weight_boundary ```
        #     --weight_initial $weight_initial ```
        #     --layers $layers ```
        #     --neurons_per_layer $neurons_per_layer ```
        #     --learning_rate $learning_rate ```
        #     --spline_degree $spline_degree ```
        #     --one_dimension ```
        #     --optimize_test_function ```
        #     --pinn_is_solution ```
        #     --save" | Out-String
        # exit 0

        ### basic case where pinn is a solution ###
        
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
            --pinn_is_solution `
            --save

        ### pinn is still a solution but we are using uneven distribution of points in X domain ###
        # python main_solution.py `
        #     --n_points_x $n_points_x `
        #     --epochs $epochs `
        #     --eps_interior $eps_interior `
        #     --weight_interior $weight_interior `
        #     --weight_boundary $weight_boundary `
        #     --weight_initial $weight_initial `
        #     --layers $layers `
        #     --neurons_per_layer $neurons_per_layer `
        #     --learning_rate $learning_rate `
        #     --spline_degree $spline_degree `
        #     --one_dimension `
        #     --pinn_is_solution `
        #     -u `
        #     --save

        ### pinn tries to learn coefficients of bsplines ###
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
            --splines `
            --save

    }
        ### pinn tries to learn coefficients of bsplines, uneven points ###
}

Write-output "Adam learns coeffs of bsplines, even points"
foreach ($eps_interior in $eps_interior_values) {
    foreach ($n_points_x in $n_points_x_values) {
        write-output "Eps interior: $eps_interior"
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
            --splines `
            --save
    }
}

Write-output "Adam learns coeffs of bsplines, uneven points"
foreach ($eps_interior in $eps_interior_values) {
    foreach ($n_points_x in $n_points_x_values) {
        write-output "Eps interior: $eps_interior"
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
            --splines `
            -u `
            --save
    }
}