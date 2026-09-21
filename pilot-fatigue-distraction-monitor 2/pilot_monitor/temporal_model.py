"""Per-pilot GRU temporal anomaly model.

New to this project, and deliberately narrower in scope than "a GRU that
classifies fatigue/distraction" -- see the design document (Part 5) for why
a supervised classifier was rejected: there is no fatigue-labeled pilot
video to train one on, and a classifier trained on too little data
memorizes subjects rather than learning fatigue.

Instead, this is a **GRU autoencoder trained only on one pilot's own
normal-condition calibration recording**. It needs no fatigue/distraction
labels at all -- only "this window was recorded while the pilot was alert
and on-task," which a short guided calibration session already guarantees
by construction. At inference time, a window of the live, personally
normalized feature stream is scored by how well the trained autoencoder can
reconstruct it: a window that looks like the pilot's own normal behaviour
reconstructs well (low error); a window that doesn't (unusual combinations
or dynamics of EAR/gaze/pose/mouth over the last N seconds) reconstructs
poorly. That reconstruction error, itself turned into a robust z-score
against the *training* error distribution, is the model's anomaly score --
one more input to ``risk.py``, alongside (not instead of) the event
statistics from ``events.py``.

This module requires PyTorch (declared in ``pyproject.toml``). It is
intentionally optional at the system level: everything in ``events.py`` /
``risk.py`` works without it, so a missing or untrained model degrades the
system to its statistical baseline rather than breaking it.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where torch is absent
    TORCH_AVAILABLE = False

FEATURE_ORDER = ("ear", "gaze_dx", "gaze_dy", "mar", "roll", "pitch", "yaw")


def zscores_to_vector(zscores):
    """Fixed-order feature vector from a ``normalization`` zscores dict.

    Missing/``None`` entries become 0.0 (i.e. "at this pilot's median"),
    which is a deliberately conservative choice: an unmeasured feature
    should not, by itself, look anomalous.
    """
    return np.array(
        [zscores.get(name) if zscores.get(name) is not None else 0.0 for name in FEATURE_ORDER],
        dtype=np.float32,
    )


def make_windows(sequence, window_len, stride=1):
    """Slice a (T, D) sequence into overlapping (window_len, D) windows."""
    sequence = np.asarray(sequence, dtype=np.float32)
    n = sequence.shape[0]
    return np.stack(
        [sequence[i : i + window_len] for i in range(0, n - window_len + 1, stride)]
    )


if TORCH_AVAILABLE:

    class GRUAutoencoder(nn.Module):
        """Encode a feature-vector sequence, then reconstruct it.

        A single-layer GRU encodes the window into its final hidden state;
        that hidden state seeds a second GRU which reconstructs the
        sequence step by step. Kept small (default 16 hidden units)
        deliberately -- a calibration recording from one pilot is at most a
        few thousand frames, and an over-parameterized model would simply
        memorize it rather than learn the pilot's normal dynamics.
        """

        def __init__(self, input_dim=len(FEATURE_ORDER), hidden_dim=16):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.decoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.output_layer = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            # x: (batch, window_len, input_dim)
            _, hidden = self.encoder(x)
            # Teacher-force the decoder with a zero-shifted version of the
            # input (standard sequence-autoencoder setup): reconstruct step
            # t from the window's own dynamics up to t-1, seeded by the
            # encoder's summary of the whole window.
            decoder_input = torch.zeros_like(x)
            decoder_input[:, 1:, :] = x[:, :-1, :]
            decoded, _ = self.decoder(decoder_input, hidden)
            return self.output_layer(decoded)


@dataclass
class TrainedTemporalModel:
    """A fitted GRU autoencoder plus the training error distribution needed
    to turn a new window's reconstruction error into an anomaly z-score."""

    pilot_id: str
    window_len: int
    hidden_dim: int
    error_median: float
    error_mad: float
    state_dict_path: str

    def anomaly_zscore(self, model, window):
        """Return a robust z-score of reconstruction error for one window.

        ``window`` is a (window_len, len(FEATURE_ORDER)) array of
        personally-normalized feature vectors (see ``zscores_to_vector``).
        """
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(window[None, :, :])
            reconstruction = model(x)
            error = float(torch.mean((reconstruction - x) ** 2))
        scale = max(self.error_mad * 1.4826, 1e-6)
        return (error - self.error_median) / scale

    def to_dict(self):
        return {
            "pilot_id": self.pilot_id,
            "window_len": self.window_len,
            "hidden_dim": self.hidden_dim,
            "error_median": self.error_median,
            "error_mad": self.error_mad,
            "state_dict_path": self.state_dict_path,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


def train_pilot_model(
    pilot_id,
    normal_sequence,
    output_dir,
    window_len=30,
    hidden_dim=16,
    epochs=60,
    lr=1e-3,
):
    """Train a per-pilot GRU autoencoder on that pilot's calibration data.

    Parameters
    ----------
    normal_sequence : np.ndarray, shape (T, len(FEATURE_ORDER))
        Personally-normalized feature vectors from a calibration recording
        of *normal, alert* behaviour only. No fatigue/distraction labels
        are needed or used.
    output_dir : str or Path
        Directory to write ``{pilot_id}_gru.pt`` (model weights) and
        ``{pilot_id}_gru.json`` (metadata: window length, error stats) to.

    Returns
    -------
    TrainedTemporalModel

    Raises
    ------
    RuntimeError
        If PyTorch is not installed, or if the calibration recording is too
        short to form even one training window.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required to train the temporal model. Install project "
            "dependencies with `uv sync` (torch is declared in pyproject.toml)."
        )

    windows = make_windows(normal_sequence, window_len)
    if len(windows) < 4:
        raise RuntimeError(
            f"Calibration recording produced only {len(windows)} training "
            f"window(s) at window_len={window_len}; need at least 4. Record "
            "a longer calibration session or reduce --window-len."
        )

    model = GRUAutoencoder(hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    x = torch.from_numpy(windows)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        reconstruction = model(x)
        loss = loss_fn(reconstruction, x)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        reconstruction = model(x)
        per_window_error = torch.mean((reconstruction - x) ** 2, dim=(1, 2)).numpy()

    error_median = float(np.median(per_window_error))
    error_mad = float(np.median(np.abs(per_window_error - error_median)))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path = output_dir / f"{pilot_id}_gru.pt"
    torch.save(model.state_dict(), state_dict_path)

    trained = TrainedTemporalModel(
        pilot_id=pilot_id,
        window_len=window_len,
        hidden_dim=hidden_dim,
        error_median=error_median,
        error_mad=error_mad,
        state_dict_path=str(state_dict_path),
    )
    (output_dir / f"{pilot_id}_gru.json").write_text(
        json.dumps(trained.to_dict(), indent=2), encoding="utf-8"
    )
    return trained


def load_pilot_model(pilot_id, model_dir):
    """Load a previously-trained per-pilot model and its GRUAutoencoder."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to load the temporal model.")
    model_dir = Path(model_dir)
    meta = TrainedTemporalModel.from_dict(
        json.loads((model_dir / f"{pilot_id}_gru.json").read_text(encoding="utf-8"))
    )
    model = GRUAutoencoder(hidden_dim=meta.hidden_dim)
    model.load_state_dict(torch.load(meta.state_dict_path, weights_only=True))
    model.eval()
    return meta, model
