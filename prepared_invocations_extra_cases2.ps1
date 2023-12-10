$epochs = 100000
$layers = 4
$neurons_per_layer = 20
$spline_degree = 3
$weight_interior = 1.0
$weight_boundary = 1.0
$weight_initial = 0.0

$learning_rate = 0.00125 # original learning rate
$eps_interior = 0.001


# 1. Proszę wziąść 10 razy mniejszy learning rate i uruchomić ten przykład
# jeszcze raz (wykres - ostatni wiersz z 6 i 7)

# $learning_rate = 0.000125
# foreach ($n_points_x in 100, 1000) {
#     # Perform some action here with $i
#     Write-Host "10x smaller learning rate, $n_points_x points."

#     python main_solution.py `
#         --n_points_x $n_points_x `
#         --epochs $epochs `
#         --eps_interior $eps_interior `
#         --weight_interior $weight_interior `
#         --weight_boundary $weight_boundary `
#         --weight_initial $weight_initial `
#         --layers $layers `
#         --neurons_per_layer $neurons_per_layer `
#         --learning_rate $learning_rate `
#         --spline_degree $spline_degree `
#         --pinn_is_solution `
#         --save `
#         --one_dimension `
#         --uneven_distribution
# }


# 2. Proszę wziąść 100 razy mniejszy learning rate i uruchomić ten
# przykład jeszcze raz (wykres - ostatni wiersz z 6 i 7)

$learning_rate = 0.0000125
foreach ($n_points_x in 100, 1000) {
    # Perform some action here with $i
    Write-Host "10x smaller learning rate, $n_points_x points."

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

# 3. Proszę wziąść 10 razy wiekszy learning rate i uruchomić ten przykład
# jeszcze raz (wykres - ostatni wiersz z 6 i 7)

# $learning_rate = 0.0125
# foreach ($n_points_x in 100, 1000) {
#     # Perform some action here with $i
#     Write-Host "10x smaller learning rate, $n_points_x points."

#     python main_solution.py `
#         --n_points_x $n_points_x `
#         --epochs $epochs `
#         --eps_interior $eps_interior `
#         --weight_interior $weight_interior `
#         --weight_boundary $weight_boundary `
#         --weight_initial $weight_initial `
#         --layers $layers `
#         --neurons_per_layer $neurons_per_layer `
#         --learning_rate $learning_rate `
#         --spline_degree $spline_degree `
#         --pinn_is_solution `
#         --save `
#         --one_dimension `
#         --uneven_distribution
# }

# later
# 4. Wybierz prosze najlepszy z case'ów 1-3 i dla niego naucz na 2 x
# wiekszej liczbie epok
# 5. Spróbuj proszę podmienic ADAM. Czy w bibliotece jest cos dostepne
# innego?
# Podobno bardzo dobry jest ADAMW lub ADAMX, oraz Lion
# A jesli nie ma to cos innego (nawet SGD)
# (to nie jest tak naprawde po to zeby to rozwiazac, tylko zeby
# odpowiedziec recenztowi)