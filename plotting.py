def plot_solution(pinn: PINN, x: torch.Tensor, t: torch.Tensor, figsize=(8, 6), dpi=100):

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x_raw = torch.unique(x).reshape(-1, 1)
    t_raw = torch.unique(t)
        
    def animate(i):

        if not i % 10 == 0:
            t_partial = torch.ones_like(x_raw) * t_raw[i]
            f_final = f(pinn, x_raw, t_partial)
            ax.clear()
            ax.plot(
                x_raw.detach().numpy(), f_final.detach().numpy(), label=f"Time {float(t[i])}"
            )
            ax.set_ylim(-1, 1)
            ax.legend()

    n_frames = t_raw.shape[0]
    return FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=False)

def plot_color(z: torch.Tensor, x: torch.Tensor, t: torch.Tensor, n_points_x, n_points_t, figsize=(8, 6), dpi=100):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    z_raw = z.detach().cpu().numpy()
    x_raw = x.detach().cpu().numpy()
    t_raw = t.detach().cpu().numpy()
    size = int(np.sqrt(z_raw.size))
    X = x_raw.reshape(n_points_t, n_points_x)
    T = t_raw.reshape(n_points_t, n_points_x)
    Z = z_raw.reshape(n_points_t, n_points_x)
    ax.set_title("PINN solution")
    ax.set_xlabel("Time")
    ax.set_ylabel("x")
    ax.set_ylabel("x")
    c = ax.pcolor(T, X, Z)
    fig.colorbar(c, ax=ax)

    return fig