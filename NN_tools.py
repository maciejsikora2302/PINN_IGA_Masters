from PINN import PINN
from differential_tools import dfdx, dfdt, f
import torch
import numpy as np
from typing import Callable, Tuple, Union
from general_parameters import logger, Color, general_parameters, OUT_DATA_FOLDER
from B_Splines import B_Splines
import tqdm
import time



def train_model(
    nn_approximator: Union[PINN,B_Splines],
    loss_fn: Callable,
    loss_fn_name: str,
    learning_rate: int = 0.01,
    max_epochs: int = 1_000,
    how_often_to_display: int = 20,
    device="cuda",
    test_function: B_Splines=None
) -> Tuple[PINN, np.ndarray]:

    if test_function is None:
        optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate)
    else:
        parameters = [
            {'params': nn_approximator.parameters()},
            {'params': test_function.parameters()}
        ]
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    loss_values = []
    lowest_current_loss = float("inf")


    for epoch in tqdm.tqdm(range(max_epochs), desc=f"{Color.BLUE}INFO -- {loss_fn_name}: {Color.RESET}", unit=" epoch"):
        try:
            loss: torch.Tensor = loss_fn(nn_approximator)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
            loss_values.append(loss.item())

            if loss_values[-1] < lowest_current_loss:
                lowest_current_loss = loss_values[-1]
                if general_parameters.save:
                    SAVE_PATH = f"{OUT_DATA_FOLDER}/model_{loss_fn_name}.pt"
                    logger.debug(f"Saving model to {Color.YELLOW}{SAVE_PATH}{Color.RESET}")
                    torch.save(nn_approximator.state_dict(), SAVE_PATH)
            # if (epoch + 1) % how_often_to_display == 0:
            #     epoch_time = time.time() - start_time
            #     time_per_epoch.append(epoch_time)
            #     avg_time_per_epoch = sum(time_per_epoch[-to_estimate:]) / min(to_estimate, len(time_per_epoch))
            #     remaining_epochs = max_epochs - epoch - 1
            #     remaining_time = avg_time_per_epoch * remaining_epochs
            #     remaining_minutes, remaining_seconds = divmod(remaining_time, 60)
            #     remaining_time_str = f"{int(remaining_minutes):02d}:{int(remaining_seconds):02d}"
                # logger.info(f"Epoch: {Color.MAGENTA}{epoch + 1}/{max_epochs}{Color.RESET} - Loss: {Color.YELLOW}{float(loss):>12f}{Color.RESET} - Time remaining: {remaining_time_str}")
                

        except KeyboardInterrupt:
            logger.info(f"Training interrupted by user at epoch {Color.RED}{epoch + 1}{Color.RESET}")
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