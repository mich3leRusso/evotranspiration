"""Masked autoencoder for the daily drivers of feature set 2 (step 1: pretraining).

Self-contained: needs only ARM_HH_2017_filled.csv in the same folder. Run with
`python autoencoder.py`. Figures go to figures/autoencoder, tables and the trained weights to
results/autoencoder, both next to this file.

Plan (one step at a time)
    1. pretrain  : masked autoencoder on the daily feature-set-2 drivers of the training
                   days only (LE is never seen). Inputs = drivers with ~30 % of the
                   features hidden + the mask; target = the full, unmasked drivers.
    2. transfer  : frozen encoder + small MLP trained to predict daily LE (MJ m-2 d-1);
                   references: same-size MLP on the raw drivers, XGBoost feature set 2
    3. attack    : per-day latent counterfactuals (Goldwasser & Hooker, 2025, Algorithm 1):
                   z0 = E(x); move z by Adam on f(D(z)) = MLP(E(D(z))) until the predicted
                   daily LE has shifted by +-ATTACK_SHIFT; the change of the decoded drivers,
                   D(z0 + delta) - D(z0) divided by each driver's observed range (their Eq. 3),
                   tells which drivers the model needs to move to change ET.
    4. shapley   : SHAP values (shap library, ExactExplainer) of the 14 drivers for the
                   autoencoder + MLP prediction on the 110 test days; background = the 255
                   training days.

Daytime / nighttime: steps 1, 2 and 4 are repeated on the half-hourly daytime and nighttime
data with their own feature set 2, the same architecture and settings, and the same
half-hourly artificial gaps as modeling.py (files with suffix _daytime / _nighttime).

Sample = one day: the 14 daily-mean feature-set-2 drivers of that date.
Data split = the same daily artificial gaps as modeling.py: 255 training days, 110 test
days. Early stopping uses inner gaps (seed 1) cut into the training days.

"Good" reconstruction, fixed before training (scored on the 110 test days):
    a) unmasked reconstruction: mean R² over the features >= 0.90
    b) each feature, masked alone, reconstructed better than its mean (R² > 0)
    reference: linear regression of each feature on the other 13 (fit on training days)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.linear_model import LinearRegression
from torch import nn
from xgboost import XGBRegressor

# Self-contained: everything is read from and written next to this file
HERE = Path(__file__).resolve().parent
DATA = HERE / "ARM_HH_2017_filled.csv"  # gap-filled half-hourly US-ARM 2017 (MDS)
FIG = HERE / "figures" / "autoencoder"
RES = HERE / "results" / "autoencoder"

# ------------------------------------------------- settings shared with the XGBoost models
UNITS = {"all": "W m⁻²", "daytime": "W m⁻²", "nighttime": "W m⁻²", "daily": "MJ m⁻² d⁻¹"}
W_TO_MJ_DAY = 0.0864              # mean W m-2 over a day -> MJ m-2 d-1
GAP_FRACTION = 0.30               # artificial gaps (Moffat 2007 gap lengths)
GAP_LENGTHS_HH = (576, 64, 8, 1)  # half-hours: long (12 d), medium, short, very short
GAP_LENGTHS_D = (12, 1)           # days
XGB_PARAMS = dict(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, random_state=0, n_jobs=-1)
# Feature set 2 per timescale (Spearman screen + RFE, Stapleton et al. 2022)
FS2 = {
    "all": "NETRAD CO2_F_MDS TS_F_MDS_4 TS_F_MDS_6 SWC_F_MDS_1 SWC_F_MDS_9 H",
    "daytime": "NETRAD CO2_F_MDS TS_F_MDS_1 TS_F_MDS_6 SWC_F_MDS_1 SWC_F_MDS_9 H RECO",
    "nighttime": "VPD_F USTAR RH NETRAD CO2_F_MDS TS_F_MDS_1 TS_F_MDS_2 TS_F_MDS_9 SWC_F_MDS_1 "
                 "SWC_F_MDS_3 SWC_F_MDS_5 SWC_F_MDS_7 G H RECO",
    "daily": "WS_F RH NETRAD TS_F_MDS_2 TS_F_MDS_6 TS_F_MDS_9 SWC_F_MDS_1 SWC_F_MDS_3 SWC_F_MDS_5 "
             "SWC_F_MDS_7 SWC_F_MDS_9 G H RECO",
}

# Plot style
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NEUTRAL = "#a3a29e"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "legend.labelcolor": INK_2, "font.size": 10,
})


def save(fig, path: Path, dpi=130):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- data loading and gaps (as in the XGBoost models)
def artificial_gaps(n: int, lengths, fraction=GAP_FRACTION, seed=0) -> np.ndarray:
    """Gap length at each position (0 = not in a gap); gaps never touch each other."""
    rng = np.random.default_rng(seed)
    gap = np.zeros(n, int)

    def place(L):
        while True:
            s = rng.integers(0, n - L + 1)
            if not gap[max(s - 1, 0): s + L + 1].any():
                gap[s: s + L] = L
                return

    for L in (L for L in lengths if L > 1):
        for _ in range(round(fraction * n / len(lengths) / L)):
            place(L)
    while (gap > 0).sum() < round(fraction * n):
        place(1)
    return gap


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily means, precipitation summed."""
    data = df.drop(columns=["TIMESTAMP_END"], errors="ignore")
    daily = data.resample("D").mean()
    if "P_F" in data:
        daily["P_F"] = data["P_F"].resample("D").sum()
    return daily


def load_datasets(path=DATA) -> dict[str, pd.DataFrame]:
    """Half-hourly and daily data per timescale, with the test-gap column `gap` (> 0 = test)."""
    hh = pd.read_csv(path, index_col=0, parse_dates=["TIMESTAMP_START"])
    hh["gap"] = artificial_gaps(len(hh), GAP_LENGTHS_HH)
    daily = to_daily(hh.drop(columns=["NIGHT", "gap"]))
    daily["LE"] *= W_TO_MJ_DAY
    daily["gap"] = artificial_gaps(len(daily), GAP_LENGTHS_D)
    return {"all": hh, "daytime": hh[hh.NIGHT == 0], "nighttime": hh[hh.NIGHT == 1], "daily": daily}


def time_axis(t: str) -> str:
    return "daily" if t == "daily" else "all"


def scores(obs: pd.Series, pred) -> dict:
    """Skill inside the gaps, in the units of LE."""
    err = np.asarray(pred) - obs
    return {"n_test": len(obs),
            "R2": 1 - (err ** 2).sum() / ((obs - obs.mean()) ** 2).sum(),
            "RMSE": np.sqrt((err ** 2).mean()), "MAE": err.abs().mean(), "bias": err.mean()}

# Architecture and training (fixed, untuned)
LATENT, HIDDEN = 8, 32  # 8 linear PCs explain 98.5 % of the variance of the 14 drivers (6 failed criterion a)
MASK_RATE = 0.3         # probability that a feature is hidden in a training sample
LR, WEIGHT_DECAY, BATCH = 1e-3, 1e-4, 32
MAX_EPOCHS, PATIENCE = 3000, 200
SEED = 0
GOOD_R2 = 0.90          # criterion a)

# Latent counterfactuals
ATTACK_SHIFT = 3.0      # MJ m-2 d-1 = 1 SD of training daily LE (~1.2 mm of water per day)
ATTACK_LR, ATTACK_MAX_STEPS = 0.01, 3000
OUT_OF_RANGE_TOL = 0.05  # decoded driver flagged if beyond the training range by > 5 % of it


# -------------------------------------------------------------------------------- data
def suffix(t: str) -> str:
    """File-name suffix: none for the daily model (its original file names), _<t> otherwise."""
    return "" if t == "daily" else f"_{t}"


def rows(t: str) -> str:
    return "days" if t == "daily" else "half-hours"


def features(t="daily") -> list[str]:
    """Feature set 2 of timescale t (RFE selection of the XGBoost models)."""
    return FS2[t].split()


def load(t="daily"):
    """(dataframe of timescale t, features, standardisation stats from the training rows)."""
    df = load_datasets()[t]
    feats = features(t)
    tr = df[df.gap == 0]
    stats = {"mean": tr[feats].mean(), "std": tr[feats].std()}
    return df, feats, stats


def to_tensor(df, feats, stats) -> torch.Tensor:
    return torch.tensor(((df[feats] - stats["mean"]) / stats["std"]).to_numpy(), dtype=torch.float32)


# ------------------------------------------------------------------------------- model
class MaskedAE(nn.Module):
    """Encoder sees the masked drivers plus the mask; decoder rebuilds all drivers."""

    def __init__(self, n_features: int, latent=LATENT, hidden=HIDDEN):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2 * n_features, hidden), nn.GELU(), nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, n_features))

    def encode(self, x, mask=None):
        """mask: 1 = hidden. Hidden values are set to 0 (= the training mean)."""
        mask = torch.zeros_like(x) if mask is None else mask
        return self.encoder(torch.cat([x * (1 - mask), mask], dim=1))

    def forward(self, x, mask=None):
        return self.decoder(self.encode(x, mask))


def random_mask(shape, rate, gen) -> torch.Tensor:
    """Bernoulli mask with at least one hidden feature per row."""
    m = (torch.rand(shape, generator=gen) < rate).float()
    empty = m.sum(1) == 0
    m[empty, torch.randint(shape[1], (int(empty.sum()),), generator=gen)] = 1
    return m


def train(model, params, n, batch_loss, epoch_log, gen):
    """Adam on `params` with early stopping on the validation loss; restores the best weights.

    batch_loss(idx) -> loss of one minibatch of the n training rows;
    epoch_log()     -> {"train_loss", "val_loss"} after each epoch (run without gradients).
    """
    opt = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    best, best_state, wait, history = np.inf, None, 0, []
    for epoch in range(MAX_EPOCHS):
        model.train()
        for idx in torch.randperm(n, generator=gen).split(BATCH):
            loss = batch_loss(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            log = epoch_log()
        history.append({"epoch": epoch, **log})
        if log["val_loss"] < best - 1e-5:
            best, best_state, wait = log["val_loss"], {k: v.clone() for k, v in model.state_dict().items()}, 0
        elif (wait := wait + 1) >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return pd.DataFrame(history)


def fit_val_split(df, t="daily") -> np.ndarray:
    """Inner validation rows (seed-1 artificial gaps on the full time axis) among the training rows."""
    full = load_datasets()[time_axis(t)].index  # day/night are subsets of the half-hourly axis
    lengths = GAP_LENGTHS_D if t == "daily" else GAP_LENGTHS_HH
    inner = pd.Series(artificial_gaps(len(full), lengths, seed=1), index=full).loc[df.index]
    return inner.to_numpy()[df.gap.to_numpy() == 0] > 0


# ------------------------------------------------------------------------ 1. pretrain
def pretrain(t="daily"):
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    df, feats, stats = load(t)
    tr = df[df.gap == 0]
    inner = fit_val_split(df, t)
    x_fit, x_val = to_tensor(tr[~inner], feats, stats), to_tensor(tr[inner], feats, stats)
    m_val = random_mask(x_val.shape, MASK_RATE, gen)  # fixed validation masks
    mse = nn.MSELoss()

    def batch_loss(idx):  # random masks on the input, full input as target
        xb = x_fit[idx]
        return mse(model(xb, random_mask(xb.shape, MASK_RATE, gen)), xb)

    def epoch_log():
        val = mse(model(x_val, m_val), x_val).item()
        return {"train_loss": mse(model(x_fit, random_mask(x_fit.shape, MASK_RATE, gen)), x_fit).item(),
                "val_loss": val}

    model = MaskedAE(len(feats))
    history = train(model, model.parameters(), len(x_fit), batch_loss, epoch_log, gen)

    RES.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RES / f"ae_pretrained{suffix(t)}.pt")
    (RES / f"ae_config{suffix(t)}.json").write_text(json.dumps(
        {"features": feats, "latent": LATENT, "hidden": HIDDEN, "mean": stats["mean"].tolist(),
         "std": stats["std"].tolist(), "best_epoch": int(history.val_loss.idxmin())}, indent=2))
    history.to_csv(RES / f"pretrain_history{suffix(t)}.csv", index=False)

    table = evaluate_reconstruction(model, df, feats, stats, t)
    plot_pretrain(table, history, t, int((df.gap > 0).sum()))
    return model, table


def _r2(obs, pred) -> np.ndarray:
    return 1 - ((pred - obs) ** 2).sum(0) / ((obs - obs.mean(0)) ** 2).sum(0)


def evaluate_reconstruction(model, df, feats, stats, t="daily") -> pd.DataFrame:
    """R² per feature on the test rows: unmasked, each feature masked alone, linear reference."""
    tr, te = df[df.gap == 0], df[df.gap > 0]
    x = to_tensor(te, feats, stats)
    obs = x.numpy()
    with torch.no_grad():
        full = model(x).numpy()
        masked = np.column_stack([model(x, torch.zeros_like(x).index_fill_(1, torch.tensor([j]), 1))[:, j].numpy()
                                  for j in range(len(feats))])
    linear = np.column_stack([LinearRegression().fit(tr[feats].drop(columns=f), tr[f]).predict(te[feats].drop(columns=f))
                              for f in feats])
    linear = (linear - stats["mean"].to_numpy()) / stats["std"].to_numpy()
    table = pd.DataFrame({"R2_unmasked": _r2(obs, full), "R2_masked_alone": _r2(obs, masked),
                          "R2_linear_from_others": _r2(obs, linear)}, index=feats).round(3)
    table.to_csv(RES / f"pretrain_reconstruction{suffix(t)}.csv")
    a = table.R2_unmasked.mean() >= GOOD_R2
    b = (table.R2_masked_alone > 0).all()
    print(table.to_string(), f"\n\nmean R² unmasked {table.R2_unmasked.mean():.3f} "
          f"(criterion a: >= {GOOD_R2} -> {'PASS' if a else 'FAIL'})"
          f"\nmasked alone > 0 for every feature (criterion b) -> {'PASS' if b else 'FAIL'}")
    return table


def plot_pretrain(table, history, t="daily", n_test=110):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [3, 1]})
    y = np.arange(len(table))[::-1]
    ax.barh(y + 0.2, table.R2_linear_from_others.clip(lower=0), height=0.38, color=NEUTRAL,
            label=f"masked: linear regression on the other {len(table) - 1} (reference)")
    ax.barh(y - 0.2, table.R2_masked_alone.clip(lower=0), height=0.38, color=CAT[0],
            label="masked: autoencoder (feature hidden alone)")
    ax.plot(table.R2_unmasked, y, "D", color=CAT[1], ms=6, label="unmasked: autoencoder reconstruction")
    ax.axvline(GOOD_R2, color=INK_2, lw=0.8, ls=":")
    ax.set_yticks(y, table.index)
    ax.set_xlim(0, 1.02)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"R² on the {n_test} test {rows(t)}")
    ax.set_title("Reconstruction of each driver")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=2, fontsize=8.5)

    ax2.plot(history.epoch, history.train_loss, color=NEUTRAL, lw=1, label="training (fit days)")
    ax2.plot(history.epoch, history.val_loss, color=CAT[0], lw=1.5, label="validation (inner gaps)")
    ax2.axvline(history.val_loss.idxmin(), color=INK_2, lw=0.8, ls=":")
    ax2.set_yscale("log")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("masked reconstruction MSE (standardised)")
    ax2.set_title("Training curve (dotted: best epoch)")
    ax2.legend(fontsize=8.5)
    fig.suptitle(f"Masked autoencoder pretraining, {t} feature set 2 ({len(table)} drivers, latent {LATENT}, "
                 f"mask rate {MASK_RATE:.0%}), training {rows(t)} only", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, FIG / f"1_pretrain_reconstruction{suffix(t)}.png")


# ------------------------------------------------------------------ 2. transfer learning
def load_pretrained(n_features: int, t="daily") -> MaskedAE:
    model = MaskedAE(n_features)
    model.load_state_dict(torch.load(RES / f"ae_pretrained{suffix(t)}.pt"))
    return model.eval()


def mlp(n_in: int) -> nn.Sequential:
    """Regression head; the same size for the latent and the raw-driver versions."""
    return nn.Sequential(nn.Linear(n_in, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))


def fit_head(head, z_fit, y_fit, z_val, y_val, gen):
    """Train `head` on standardised LE; inputs are fixed tensors (frozen encoder or raw)."""
    mse = nn.MSELoss()
    history = train(head, head.parameters(), len(z_fit),
                    lambda idx: mse(head(z_fit[idx]), y_fit[idx]),
                    lambda: {"train_loss": mse(head(z_fit), y_fit).item(),
                             "val_loss": mse(head(z_val), y_val).item()}, gen)
    return history


def transfer(t="daily"):
    """Frozen pretrained encoder + MLP head -> LE of timescale t; compared on the same test rows."""
    df, feats, stats = load(t)
    tr, te = df[df.gap == 0], df[df.gap > 0]
    inner = fit_val_split(df, t)
    y_mean, y_std = tr.LE.mean(), tr.LE.std()
    y = torch.tensor(((tr.LE - y_mean) / y_std).to_numpy(), dtype=torch.float32)[:, None]

    ae = load_pretrained(len(feats), t)
    for p in ae.parameters():
        p.requires_grad_(False)  # the encoder is frozen: only the head learns
    x_tr, x_te = to_tensor(tr, feats, stats), to_tensor(te, feats, stats)
    with torch.no_grad():
        z_tr, z_te = ae.encode(x_tr), ae.encode(x_te)

    preds, histories = {}, {}
    for name, (a, b) in {"autoencoder + MLP": (z_tr, z_te), "MLP on raw drivers": (x_tr, x_te)}.items():
        torch.manual_seed(SEED)  # same initialisation and batch order for every head
        head = mlp(a.shape[1])
        gen = torch.Generator().manual_seed(SEED)
        histories[name] = fit_head(head, a[~inner], y[~inner], a[inner], y[inner], gen)
        with torch.no_grad():
            preds[name] = head(b).squeeze(1).numpy() * y_std + y_mean
        if name == "autoencoder + MLP":
            torch.save(head.state_dict(), RES / f"mlp_head{suffix(t)}.pt")
            (RES / f"mlp_head_config{suffix(t)}.json").write_text(json.dumps({"y_mean": y_mean, "y_std": y_std}))

    xgb = XGBRegressor(**XGB_PARAMS).fit(tr[feats], tr.LE)  # reference: XGBoost on the same features
    preds["XGBoost feature set 2"] = xgb.predict(te[feats])
    out = te[["LE", "gap"]].rename(columns={"LE": "obs"}).assign(**preds)
    out.to_csv(RES / f"transfer_predictions{suffix(t)}.csv")
    table = pd.DataFrame([{"model": m, **scores(out.obs, out[m])} for m in preds]).round(3)
    table.to_csv(RES / f"transfer_scores{suffix(t)}.csv", index=False)
    print(table.to_string(index=False))
    plot_transfer(out, table, t)
    return table


def plot_transfer(out, table, t="daily"):
    colors = {"autoencoder + MLP": CAT[0], "MLP on raw drivers": CAT[4], "XGBoost feature set 2": NEUTRAL}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axes[0]
    s = table.set_index("model").loc["autoencoder + MLP"]
    ax.scatter(out.obs, out["autoencoder + MLP"], s=16, color=CAT[0], lw=0, alpha=0.85)
    hi = max(out.obs.max(), out["autoencoder + MLP"].max())
    ax.plot([0, hi], [0, hi], color=INK_2, lw=1, ls="--")
    ax.set_aspect("equal")
    ax.set_xlabel(f"observed LE ({UNITS[t]})")
    ax.set_ylabel(f"predicted LE ({UNITS[t]})")
    ax.set_title(f"autoencoder + MLP\nR² {s.R2:.2f} · RMSE {s.RMSE:.2f} · bias {s.bias:+.2f} · n {int(s.n_test)}",
                 fontsize=10)
    for ax, (metric, label, fmt) in zip(axes[1:], (("R2", "R² on the test days", "{:.3f}"),
                                                   ("RMSE", f"RMSE ({UNITS[t]}), lower is better", "{:.2f}"))):
        v = table.set_index("model").loc[list(colors), metric]
        bars = ax.bar(range(len(v)), v, color=list(colors.values()), width=0.6)
        ax.bar_label(bars, [fmt.format(b) for b in v], fontsize=9, color=INK_2, padding=2)
        ax.set_xticks(range(len(v)), [m.replace(" + ", " +\n").replace(" on ", "\non ").replace(" feature", "\nfeature")
                                      for m in v.index], fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.margins(y=0.12)
    fig.suptitle(f"Transfer learning: frozen masked-autoencoder encoder + MLP head predicting {t} LE "
                 f"(trained on the training {rows(t)}, scored on the {len(out)} test {rows(t)})",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, FIG / f"2_transfer_learning{suffix(t)}.png")


# --------------------------------------------------------- 3. latent counterfactuals
def load_head(t="daily"):
    head = mlp(LATENT)
    head.load_state_dict(torch.load(RES / f"mlp_head{suffix(t)}.pt"))
    cfg = json.loads((RES / f"mlp_head_config{suffix(t)}.json").read_text())
    return head.eval(), cfg["y_mean"], cfg["y_std"]


def latent_counterfactuals(f, z0, target, direction, lr=ATTACK_LR, max_steps=ATTACK_MAX_STEPS):
    """Algorithm 1 of Goldwasser & Hooker (2025), batched over days.

    Each row of z moves along +-grad f until f(z) crosses its target (direction +1: up,
    -1: down); the latent at the first crossing is kept. Rows that never cross keep their
    last iterate and are reported as not reached.
    """
    z = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    z_end, steps = z0.clone(), torch.full((len(z0),), max_steps)
    done = torch.zeros(len(z0), dtype=torch.bool)
    for step in range(max_steps):
        pred = f(z)
        new = ((direction * (pred - target)) >= 0) & ~done
        z_end[new], steps[new] = z.detach()[new], step
        done |= new
        if done.all():
            break
        opt.zero_grad()
        (-(direction * pred[~done]).sum()).backward()
        opt.step()
    z_end[~done] = z.detach()[~done]
    return z_end, steps, done


def attack():
    """Per-day latent vectors delta that shift predicted daily LE by +-ATTACK_SHIFT."""
    daily, feats, stats = load("daily")
    tr = daily[daily.gap == 0]
    ae, (head, y_mean, y_std) = load_pretrained(len(feats)), load_head()
    for p in [*ae.parameters(), *head.parameters()]:
        p.requires_grad_(False)

    def f(z):  # predicted daily LE (MJ m-2 d-1) of the decoded input, as in the paper
        return head(ae.encode(ae.decoder(z))).squeeze(1) * y_std + y_mean

    def decode(z):  # drivers in physical units
        with torch.no_grad():
            return ae.decoder(z).numpy() * stats["std"].to_numpy() + stats["mean"].to_numpy()

    with torch.no_grad():
        z0 = ae.encode(to_tensor(daily, feats, stats))
        pred0 = f(z0)
    x0 = decode(z0)
    lo, hi = tr[feats].min().to_numpy(), tr[feats].max().to_numpy()
    diam = hi - lo

    rows = []
    for direction, name in ((1, "up"), (-1, "down")):
        days = torch.ones(len(z0), dtype=torch.bool) if direction > 0 else pred0 >= ATTACK_SHIFT
        target = pred0[days] + direction * ATTACK_SHIFT
        z_end, steps, done = latent_counterfactuals(f, z0[days], target, direction)
        with torch.no_grad():
            pred_end = f(z_end)
        x_end = decode(z_end)
        phi = (x_end - x0[days.numpy()]) / diam  # Eq. 3: change in units of the observed range
        outside = ((x_end < lo - OUT_OF_RANGE_TOL * diam) | (x_end > hi + OUT_OF_RANGE_TOL * diam)).any(1)
        tab = pd.DataFrame(phi, columns=feats, index=daily.index[days.numpy()])
        tab.insert(0, "out_of_range", outside)
        tab.insert(0, "delta_norm", (z_end - z0[days]).norm(dim=1).numpy())
        tab.insert(0, "steps", steps.numpy())
        tab.insert(0, "reached", done.numpy())
        tab.insert(0, "LE_pred_end", pred_end.numpy())
        tab.insert(0, "LE_pred_start", pred0[days].numpy())
        tab.insert(0, "direction", name)
        rows.append(tab)
    per_day = pd.concat(rows).round(4)
    per_day.to_csv(RES / "attack_per_day.csv")

    ok = per_day[per_day.reached & ~per_day.out_of_range]  # counterfactuals kept for the summary
    sens = ok.groupby("direction")[feats].agg(["mean", lambda v: v.abs().mean()]).T
    sens.index = pd.MultiIndex.from_tuples([(f, "mean" if s == "mean" else "mean_abs") for f, s in sens.index])
    sens = sens.unstack().round(4)
    sens.to_csv(RES / "attack_sensitivity.csv")
    summary = per_day.groupby("direction").agg(days=("reached", "size"), reached=("reached", "sum"),
                                               out_of_range=("out_of_range", "sum"),
                                               median_steps=("steps", "median"),
                                               median_delta_norm=("delta_norm", "median"))
    print(summary.to_string(), "\n\nmean change of each decoded driver (fraction of its observed range)\n",
          sens.to_string())
    plot_attack(ok, feats, summary)
    return per_day, sens


def plot_attack(ok, feats, summary):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    labels = {"up": "ORANGE: model pushed to predict MORE ET; bar = how each driver changes to raise ET",
              "down": "BLUE: model pushed to predict LESS ET; bar = how each driver changes to lower ET"}
    for ax, (name, color) in zip(axes, (("up", CAT[1]), ("down", CAT[0]))):
        d = ok[ok.direction == name][feats]
        order = d.mean().abs().sort_values().index  # ranked by bar length, largest change on top
        d = d[order]
        y = np.arange(len(order))
        ax.barh(y, d.mean(), color=color, height=0.6)
        ax.hlines(y, d.quantile(0.25), d.quantile(0.75), color=INK_2, lw=1.2)  # interquartile range
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.set_yticks(y, order)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        s = summary.loc[name]
        sign = "+" if name == "up" else "−"
        ax.set_title(f"{labels[name]}\npredicted ET {sign}{ATTACK_SHIFT:.0f} MJ m⁻² d⁻¹ "
                     f"({len(d)} days kept of {int(s.days)}: {int(s.reached)} reached, {int(s.out_of_range)} out of range)",
                     fontsize=10)
        ax.set_xlabel("change of the decoded driver (fraction of its observed range); line = IQR")
    fig.suptitle("Latent counterfactuals (Goldwasser & Hooker, 2025): which drivers the autoencoder + MLP model "
                 "has to move to change daily ET\nper-day latent vector δ found by gradient ascent/descent on "
                 "f(D(z)); bars = mean of D(z₀ + δ) − D(z₀) over days", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, FIG / "3_latent_counterfactuals.png")


# ------------------------------------------------------------------ 4. Shapley values
def shapley(t="daily"):
    """SHAP values of the feature-set-2 drivers for the autoencoder + MLP prediction of LE (test rows).

    shap library (Lundberg & Lee, 2017), shap.ExactExplainer with an Independent masker.
    Background: all 255 training days (daily); the library default of 100 training rows
    (sampled by the masker) for daytime and nighttime.
    """
    df, feats, stats = load(t)
    tr, te = df[df.gap == 0], df[df.gap > 0]
    ae, (head, y_mean, y_std) = load_pretrained(len(feats), t), load_head(t)

    def f(x):  # full model on standardised drivers (numpy) -> LE in the units of timescale t
        with torch.no_grad():
            return (head(ae.encode(torch.tensor(x, dtype=torch.float32))).squeeze(1) * y_std + y_mean).numpy()

    background = to_tensor(tr, feats, stats).numpy()
    n_bg = len(background) if t == "daily" else 100
    masker = shap.maskers.Independent(background, max_samples=n_bg)
    x_te = to_tensor(te, feats, stats).numpy()
    explanation = shap.ExactExplainer(f, masker)(x_te)
    phi, base = explanation.values, explanation.base_values
    gap = np.abs(phi.sum(1) + base - f(x_te)).max()
    values = pd.DataFrame(phi, index=te.index, columns=feats)
    values.round(5).to_csv(RES / f"shap_ae_{t}.csv")
    importance = values.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").round(4)
    importance.to_csv(RES / f"shap_importance_ae{suffix(t)}.csv")
    print(f"{t}: base value (mean prediction on the background) {base[0]:.3f} {UNITS[t]}; "
          f"max |sum(phi) + base - f(x)| = {gap:.1e}\n\nmean |SHAP| ({UNITS[t]})\n", importance.to_string())
    plot_shapley(importance, t, len(te), n_bg)
    return values, importance


def plot_shapley(importance, t="daily", n_test=110, n_bg=255):
    imp = importance.sort_values()
    share = 100 * imp / imp.sum()
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(imp) + 0.8))
    ax.barh(imp.index, imp, color=CAT[0], height=0.6)
    fmt = ".2f" if t == "daily" else ".1f"
    for y, (v, pc) in enumerate(zip(imp, share)):
        ax.text(v, y, f" {v:{fmt}} ({pc:.0f}%)", va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, imp.max() * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"mean |SHAP| ({UNITS[t]})")
    ax.set_title(f"Feature importance, autoencoder + MLP ({t} LE): mean |SHAP value| on the {n_test} test {rows(t)}\n"
                 f"shap.ExactExplainer, background = {n_bg} training {rows(t)}; % = share of total",
                 fontsize=10)
    fig.tight_layout()
    save(fig, FIG / f"5_shap_importance{suffix(t)}.png")


def main():
    torch.set_num_threads(4)
    pretrain()
    transfer()
    attack()
    shapley()
    for t in ("daytime", "nighttime"):  # same architecture and settings on the half-hourly day/night data
        pretrain(t)
        transfer(t)
        shapley(t)


if __name__ == "__main__":
    main()
