$epochs = 150_000
$layers = 4
$neurons_per_layer = 20
$learning_rate = 0.00125
$spline_degree = 3
$weight_interior = 1.0
$weight_boundary = 1.0
$weight_initial = 0.0

$eps_interior = 0.001
$n_points_x = 1000



Write-Host "Parameters: "
Write-Host "  epochs: $epochs"
Write-Host "  layers: $layers"
Write-Host "  neurons_per_layer: $neurons_per_layer"
Write-Host "  learning_rate: $learning_rate"
Write-Host "  spline_degree: $spline_degree"
Write-Host "  weight_interior: $weight_interior"
Write-Host "  weight_boundary: $weight_boundary"
Write-Host "  weight_initial: $weight_initial"
Write-Host "  "

foreach ($iteration in 1..10) {
    Write-Host "Iteration $iteration"

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
        --save `
        --one_dimension
}

Write-Host "Done. Moving to uneven distribution."

foreach ($iteration in 1..10) {
    Write-Host "Iteration $iteration"

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
        --save `
        --one_dimension `
        --uneven_distribution
}