"""Lab 7 reference solution: system identification, LQR, and model predictive control.

Run from this folder:  python lab07_optimal_control.py
It prints the results quoted on the lab page. Takes about two minutes on one core.
"""
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import gymnasium as gym
import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

t0 = time.time()

# ---------------------------------------------------------------- part 1: identify the cart-pole, then LQR
g, mc, mp, l, dt = 9.8, 1.0, 0.1, 0.5, 0.02


def cartpole(x, u):
    """Gymnasium's cart-pole dynamics (Euler, 0.02 s) with a continuous force u; x = (pos, vel, angle, ang. vel.)."""
    pos, vel, th, om = x.T
    total, pml = mc + mp, mp * l
    temp = (u + pml * om ** 2 * np.sin(th)) / total
    th_acc = (g * np.sin(th) - np.cos(th) * temp) / (l * (4 / 3 - mp * np.cos(th) ** 2 / total))
    x_acc = temp - pml * th_acc * np.cos(th) / total
    return np.stack([pos + dt * vel, vel + dt * x_acc, th + dt * om, om + dt * th_acc], -1)


def lqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def cost_of(K, A, B, Q, R):
    """Infinite-horizon cost tr(P_K) of u = -Kx from x_0 ~ N(0, I) on (A, B); infinite if unstable."""
    Acl = A - B @ K
    if np.abs(np.linalg.eigvals(Acl)).max() >= 1:
        return np.inf
    return np.trace(solve_discrete_lyapunov(Acl.T, Q + K.T @ R @ K))


eps = 1e-6
A = np.column_stack([(cartpole(eps * e, 0.0) - cartpole(-eps * e, 0.0)) / (2 * eps) for e in np.eye(4)])
B = ((cartpole(np.zeros(4), eps) - cartpole(np.zeros(4), -eps)) / (2 * eps))[:, None]
Q, R = np.diag([1.0, 0.1, 10.0, 0.1]), np.array([[0.01]])
K_true = lqr(A, B, Q, R); C_opt = cost_of(K_true, A, B, Q, R)

print("=== Part 1: least-squares identification of the cart-pole near upright, and certainty-equivalent LQR ===")
print("  N transitions   error in [A B]   error in the gain   relative excess cost on the linearization   balances from 15 degrees")
rng = np.random.default_rng(0)
for N in (10, 20, 50, 100, 1000, 10000):
    errs, gerrs, ratios, ok = [], [], [], 0
    for trial in range(20):
        X = 0.05 * rng.standard_normal((N, 4)); U = rng.standard_normal(N)            # random states and forces
        X1 = cartpole(X, U) + 0.001 * rng.standard_normal((N, 4))                     # small measurement noise
        Z = np.hstack([X, U[:, None]])
        Theta = np.linalg.lstsq(Z, X1, rcond=None)[0].T                                # [A B] by least squares
        Ah, Bh = Theta[:, :4], Theta[:, 4:]
        errs.append(np.linalg.norm(Theta - np.hstack([A, B])) / np.linalg.norm(np.hstack([A, B])))
        try:
            K = lqr(Ah, Bh, Q, R)
        except (np.linalg.LinAlgError, ValueError):
            gerrs.append(np.inf); ratios.append(np.inf); continue
        gerrs.append(np.linalg.norm(K - K_true) / np.linalg.norm(K_true))
        ratios.append(cost_of(K, A, B, Q, R) / C_opt)
        x = np.array([0, 0, np.radians(15), 0.0])
        for t in range(500):
            x = cartpole(x, float(np.clip(-(K @ x)[0], -10, 10)))
            if abs(x[2]) > np.pi / 2 or abs(x[0]) > 2.4:
                break
        ok += t == 499
    med = lambda v: np.median(v)
    print(f"  {N:13,d}   {med(errs):14.1e}   {med(gerrs):17.1e}   {med(ratios) - 1:40.1e}   {ok:15d} of 20")
print(f"  (medians over 20 data sets; time so far {time.time() - t0:.0f} s)")

# ---------------------------------------------------------------- part 2: MPC on Gymnasium's pendulum
print("\n=== Part 2: MPC with MPPI on Gymnasium's Pendulum-v1 ===")
G, M, L, DT, MAX_SPEED, MAX_TORQUE = 10.0, 1.0, 1.0, 0.05, 8.0, 2.0


def pendulum(x, u):
    """Pendulum-v1's dynamics, vectorized: x = (theta, theta_dot), u = torque."""
    u = np.clip(u, -MAX_TORQUE, MAX_TORQUE)
    om = np.clip(x[..., 1] + (3 * G / (2 * L) * np.sin(x[..., 0]) + 3 / (M * L ** 2) * u) * DT, -MAX_SPEED, MAX_SPEED)
    return np.stack([x[..., 0] + om * DT, om], -1)


def gym_cost(x, u):
    th = (x[..., 0] + np.pi) % (2 * np.pi) - np.pi
    return th ** 2 + 0.1 * x[..., 1] ** 2 + 0.001 * np.clip(u, -MAX_TORQUE, MAX_TORQUE) ** 2


env = gym.make("Pendulum-v1"); worst = 0.0; rng = np.random.default_rng(1)
for ep in range(5):
    env.reset(seed=ep); x = env.unwrapped.state.copy()
    for t in range(200):
        u = rng.uniform(-2, 2)
        env.step(np.array([u])); x = pendulum(x, u)
        worst = max(worst, np.abs(x - env.unwrapped.state).max())
print(f"  largest state difference from Gymnasium over 5 random episodes: {worst:.1e}")


def mppi_mpc(model, H=30, n=200, iters=2, sigma=1.0, lam=0.3):
    """Returns a controller: state -> torque, re-planning an H-step torque sequence with MPPI at every call."""
    plan = {"U": np.zeros(H)}
    rng = np.random.default_rng(2)

    def act(x):
        U = plan["U"]
        for _ in range(iters):
            S = np.clip(U + sigma * rng.standard_normal((n, H)), -MAX_TORQUE, MAX_TORQUE)
            xs, J = np.tile(x, (n, 1)), np.zeros(n)
            for t in range(H):
                J += gym_cost(xs, S[:, t]); xs = model(xs, S[:, t])
            w = np.exp(-(J - J.min()) / lam); w /= w.sum()
            U = np.clip(U + w @ (S - U), -MAX_TORQUE, MAX_TORQUE)
        plan["U"] = np.append(U[1:], 0.0)
        return U[0]
    act.reset = lambda: plan.update(U=np.zeros(H))
    return act


def evaluate(controller, episodes=10):
    """Average undiscounted return of Pendulum-v1 (200 steps) from Gymnasium's random initial states."""
    env, returns = gym.make("Pendulum-v1"), []
    for ep in range(episodes):
        obs, _ = env.reset(seed=100 + ep); total = 0.0
        if hasattr(controller, "reset"):
            controller.reset()
        for t in range(200):
            obs, r, term, trunc, _ = env.step(np.array([controller(env.unwrapped.state.copy())]))
            total += r
        returns.append(total)
    return np.mean(returns), np.std(returns)


rng_r = np.random.default_rng(3)
print("  average return over 10 episodes (standard deviation):")
m, s = evaluate(lambda x: rng_r.uniform(-2, 2)); print(f"    random torques:                       {m:8.1f} ({s:5.1f})")
m, s = evaluate(lambda x: 0.0); print(f"    no torque:                            {m:8.1f} ({s:5.1f})")
for H, n in [(10, 200), (20, 200), (30, 200), (30, 50), (50, 200)]:
    tt = time.time(); m, s = evaluate(mppi_mpc(pendulum, H=H, n=n))
    print(f"    MPC, horizon {H:2d}, {n:3d} samples:          {m:8.1f} ({s:5.1f})   {(time.time() - tt) / 2000 * 1000:.1f} ms per step")
print(f"  (time so far {time.time() - t0:.0f} s)")

# ---------------------------------------------------------------- part 3: MPC with a learned model
print("\n=== Part 3: MPC with models learned from 1,000 random transitions ===")
env = gym.make("Pendulum-v1"); X, U, X1 = [], [], []
obs, _ = env.reset(seed=7); rng = np.random.default_rng(4)
for t in range(1000):
    if t % 200 == 0:
        env.reset(seed=1000 + t)
    x = env.unwrapped.state.copy(); u = rng.uniform(-2, 2)
    env.step(np.array([u])); X.append(x); U.append(u); X1.append(env.unwrapped.state.copy())
X, U, X1 = np.array(X), np.array(U), np.array(X1)
dom = (X1[:, 1] - X[:, 1]) / DT                                  # observed angular acceleration


def quad(z):                                                     # 1, z_i, and z_i z_j for i <= j: 15 features
    i, j = np.triu_indices(z.shape[-1])
    return np.concatenate([np.ones(z.shape[:-1] + (1,)), z, z[..., i] * z[..., j]], -1)


def fit(features):
    Phi = features(X, U)
    w = np.linalg.lstsq(Phi, dom, rcond=None)[0]
    rmse = np.sqrt(np.mean((Phi @ w - dom) ** 2))
    def model(x, u):
        u = np.clip(u, -MAX_TORQUE, MAX_TORQUE)
        om = np.clip(x[..., 1] + (features(x, u) @ w) * DT, -MAX_SPEED, MAX_SPEED)
        return np.stack([x[..., 0] + om * DT, om], -1)
    return model, w, rmse


models = {
    "sin(theta), torque (the true form)": lambda x, u: np.stack([np.sin(x[..., 0]), u], -1),
    "theta, torque (linearized)": lambda x, u: np.stack([(x[..., 0] + np.pi) % (2 * np.pi) - np.pi, u], -1),
    "torque only (no gravity)": lambda x, u: np.stack([u], -1),
    "all quadratics in (cos, sin, speed, torque)": lambda x, u: quad(np.stack([np.cos(x[..., 0]), np.sin(x[..., 0]), x[..., 1], u], -1)),
}
for name, feats in models.items():
    model, w, rmse = fit(feats)
    m, s = evaluate(mppi_mpc(model))
    shown = np.round(w, 2) if len(w) <= 2 else f"{len(w)} weights"
    print(f"  {name:44s} {shown!s:15s} fit error {rmse:5.2f}   MPC return {m:8.1f} ({s:5.1f})")
print(f"  (the true weights are 3g/(2l) = {3 * G / (2 * L):.0f} and 3/(ml^2) = {3 / (M * L ** 2):.0f})")
print(f"\ntotal time {time.time() - t0:.0f} s")
