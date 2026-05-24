import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─────────────────────────────────────────────
# PINN — Équation de la chaleur : ∂u/∂t = α ∂²u/∂x²
# Conditions :  u(x, 0) = sin(πx)
#               u(0, t) = u(1, t) = 0
# Solution exacte : u(x,t) = sin(πx) · exp(-α·π²·t)
# ─────────────────────────────────────────────

ALPHA = 1.0   # coefficient de diffusion

torch.manual_seed(42)


# ── Réseau de neurones (dict de paramètres) ───────────────────────
def initialize_parameters(dimension):
    parameters = {}
    for layer in range(1, len(dimension)):
        fan_in = dimension[layer - 1]
        parameters['W' + str(layer)] = (
            torch.randn(dimension[layer], fan_in, dtype=torch.float64)
            * (1.0 / fan_in) ** 0.5
        ).requires_grad_(True)
        parameters['b' + str(layer)] = torch.zeros(
            (dimension[layer], 1), dtype=torch.float64, requires_grad=True
        )
    return parameters


def forward_batch(params, x, t):
    """
    Passe avant vectorisée.
    x, t : tenseurs (N,) — peuvent avoir requires_grad=True
    Retourne u : (N,)
    """
    L = len(params) // 2
    A = torch.stack([x, t])          # (2, N)
    for layer in range(1, L + 1):
        Z = params['W' + str(layer)] @ A + params['b' + str(layer)]
        A = Z if layer == L else torch.tanh(Z)
    return A.squeeze(0)              # (N,)


# ── Loss physique ─────────────────────────────────────────────────
def loss_physics(params, x_col, t_col, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc):
    """
    Utilise torch.autograd.grad avec create_graph=True pour les
    dérivées de l'EDP (rapide, reste dans le graphe de calcul).
    """
    # ── 1. Résidu de l'EDP ────────────────────────────────────────
    # x_r / t_r doivent avoir requires_grad pour les dérivées spatiales
    x_r = x_col.detach().requires_grad_(True)
    t_r = t_col.detach().requires_grad_(True)

    u_r = forward_batch(params, x_r, t_r)   # (N,)

    # Astuce : grad(sum u_i, x_r) = [∂u_1/∂x_1, ..., ∂u_N/∂x_N]
    # valide car u_i ne dépend que de (x_i, t_i) dans le batch
    du_dt  = torch.autograd.grad(u_r.sum(), t_r, create_graph=True)[0]
    du_dx  = torch.autograd.grad(u_r.sum(), x_r, create_graph=True)[0]
    du_dxx = torch.autograd.grad(du_dx.sum(), x_r, create_graph=True)[0]

    residual  = du_dt - ALPHA * du_dxx
    loss_res  = torch.mean(residual ** 2)

    # ── 2. Condition initiale : u(x, 0) = sin(πx) ────────────────
    u_pred_ic = forward_batch(params, x_ic, t_ic)
    loss_ci   = torch.mean((u_pred_ic - u_ic) ** 2)

    # ── 3. Conditions aux limites : u(0,t) = u(1,t) = 0 ─────────
    u_pred_bc = forward_batch(params, x_bc, t_bc)
    loss_cl   = torch.mean((u_pred_bc - u_bc) ** 2)

    return loss_res + loss_ci + loss_cl


# ── Entraînement ──────────────────────────────────────────────────
def train_neural_network(
    dimension, n_iterations, learning_rate=1e-3, log_interval=50
):
    parameters = initialize_parameters(dimension)
    optimizer  = torch.optim.Adam(list(parameters.values()), lr=learning_rate)
    history    = {'loss': []}
    snapshots  = []

    for iteration in tqdm(range(n_iterations), desc="Training"):
        optimizer.zero_grad()
        loss = loss_physics(
            parameters,
            x_col, t_col, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc
        )
        loss.backward()
        optimizer.step()

        if iteration % log_interval == 0:
            history['loss'].append(loss.item())
            snapshots.append((iteration, loss.item()))

    return parameters, history, snapshots


# ── Données d'entraînement ────────────────────────────────────────
N_col, N_ic, N_bc = 1000, 100, 100

x_col = torch.rand(N_col, dtype=torch.float64)
t_col = torch.rand(N_col, dtype=torch.float64)

x_ic  = torch.rand(N_ic, dtype=torch.float64)
t_ic  = torch.zeros(N_ic, dtype=torch.float64)
u_ic  = torch.sin(torch.pi * x_ic)

x_bc  = torch.cat([torch.zeros(N_bc // 2), torch.ones(N_bc // 2)]).to(torch.float64)
t_bc  = torch.rand(N_bc, dtype=torch.float64)
u_bc  = torch.zeros(N_bc, dtype=torch.float64)   # sin(0) = sin(π) = 0


# ── Lancement ─────────────────────────────────────────────────────
parameters, history, snapshots = train_neural_network(
    dimension      = [2, 32, 32, 1],
    n_iterations   = 5000,
    learning_rate  = 1e-3,
    log_interval   = 50,
)


# ── Visualisation ─────────────────────────────────────────────────
iterations_log = [s[0] for s in snapshots]
losses         = [s[1] for s in snapshots]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(
    f"PINN — Équation de la chaleur  ∂u/∂t = {ALPHA}·∂²u/∂x²", fontsize=13
)

# 1. Courbe de loss
axes[0].semilogy(iterations_log, losses, 'g-', linewidth=2)
axes[0].set_xlabel('Itération'); axes[0].set_ylabel('Loss')
axes[0].set_title('Courbe de loss'); axes[0].grid(True, alpha=0.3)

# 2. Solution PINN vs exacte à t = 0.5
with torch.no_grad():
    x_test  = torch.linspace(0, 1, 300, dtype=torch.float64)
    t_fixed = torch.full((300,), 0.5, dtype=torch.float64)
    u_pred  = forward_batch(parameters, x_test, t_fixed)
    u_exact = torch.sin(torch.pi * x_test) * torch.exp(
        -ALPHA * torch.pi ** 2 * t_fixed
    )

axes[1].plot(x_test, u_exact, 'k--', linewidth=2, label='Exacte')
axes[1].plot(x_test, u_pred,  'b-',  linewidth=2, label='PINN')
axes[1].set_xlabel('x'); axes[1].set_ylabel('u(x, t=0.5)')
axes[1].set_title('Solution à t = 0.5')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# 3. Heatmap u(x, t)
Nx, Nt = 100, 100
x_plot = torch.linspace(0, 1, Nx, dtype=torch.float64)
t_plot = torch.linspace(0, 1, Nt, dtype=torch.float64)
X, T   = torch.meshgrid(x_plot, t_plot, indexing='ij')
with torch.no_grad():
    U = forward_batch(parameters, X.flatten(), T.flatten()).reshape(Nx, Nt)

im = axes[2].pcolormesh(T.numpy(), X.numpy(), U.numpy(), cmap='hot', shading='auto')
plt.colorbar(im, ax=axes[2])
axes[2].set_xlabel('t'); axes[2].set_ylabel('x')
axes[2].set_title('u(x, t) — heatmap')

plt.tight_layout()
plt.savefig('assets/solution.png', dpi=150, bbox_inches='tight')
plt.show()

loss_final = snapshots[-1][1]
print(f"\n✅ Terminé — loss finale : {loss_final:.2e}")
print("   Figure sauvegardée dans assets/solution.png")
