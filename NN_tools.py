def train_model(
    nn_approximator: PINN,
    loss_fn: Callable,
    learning_rate: int = 0.01,
    max_epochs: int = 1_000,
    device="cpu"
) -> PINN:

    optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate)
    loss_values = []
    for epoch in range(max_epochs):

        try:

            loss: torch.Tensor = loss_fn(nn_approximator)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())
            if (epoch + 1) % 100 == 0:
                print(f"Epoch: {epoch + 1} - Loss: {float(loss):>7f}")

        except KeyboardInterrupt:
            break

    return nn_approximator, np.array(loss_values)


def check_gradient(nn_approximator: PINN, x: torch.Tensor, t: torch.Tensor) -> bool:

    eps = 1e-4
    
    dfdx_fd = (f(nn_approximator, x + eps, t) - f(nn_approximator, x - eps, t)) / (2 * eps)
    dfdx_autodiff = dfdx(nn_approximator, x, t, order=1)
    is_matching_x = torch.allclose(dfdx_fd.T, dfdx_autodiff.T, atol=1e-2, rtol=1e-2)

    dfdt_fd = (f(nn_approximator, x, t + eps) - f(nn_approximator, x, t - eps)) / (2 * eps)
    dfdt_autodiff = dfdt(nn_approximator, x, t, order=1)
    is_matching_t = torch.allclose(dfdt_fd.T, dfdt_autodiff.T, atol=1e-2, rtol=1e-2)
    
    eps = 1e-2

    d2fdx2_fd = (f(nn_approximator, x + eps, t) - 2 * f(nn_approximator, x, t) + f(nn_approximator, x - eps, t)) / (eps ** 2)
    d2fdx2_autodiff = dfdx(nn_approximator, x, t, order=2)
    is_matching_x2 = torch.allclose(d2fdx2_fd.T, d2fdx2_autodiff.T, atol=1e-2, rtol=1e-2)

    d2fdt2_fd = (f(nn_approximator, x, t + eps) - 2 * f(nn_approximator, x, t) + f(nn_approximator, x, t - eps)) / (eps ** 2)
    d2fdt2_autodiff = dfdt(nn_approximator, x, t, order=2)
    is_matching_t2 = torch.allclose(d2fdt2_fd.T, d2fdt2_autodiff.T, atol=1e-2, rtol=1e-2)
    
    return is_matching_x and is_matching_t and is_matching_x2 and is_matching_t2

def sin_act(x):
    return torch.sin(x)