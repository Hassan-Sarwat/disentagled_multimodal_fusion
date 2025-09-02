import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn


# ---------- One-label dataset ----------
class MultimodalDataset(Dataset):
    """
    total_data: np.ndarray of shape (M, N, D) or a list [X, Y, ...] where each modality is (N, Dm)
    labels: np.ndarray of shape (N,) with {0,1}
    """
    def __init__(self, total_data, labels=None):
        if isinstance(total_data, (list, tuple)):
            self.modalities = [torch.from_numpy(m).float() for m in total_data]
            n = self.modalities[0].shape[0]
            assert all(m.shape[0] == n for m in self.modalities), "All modalities need same N."
            self.num_modalities = len(self.modalities)
        else:
            td = torch.from_numpy(total_data).float()
            assert td.ndim == 3, "total_data must have shape (M, N, D)"
            self.num_modalities = td.shape[0]
            self.modalities = [td[i] for i in range(self.num_modalities)]

        self.n = self.modalities[0].shape[0]
        # in your Dataset.__init__:
        self.labels = None if labels is None else torch.from_numpy(labels).long()

    def __len__(self): return self.n

    def __getitem__(self, idx):
        xs = tuple(m[idx] for m in self.modalities)
        return xs + (self.labels[idx],) if self.labels is not None else xs

    def sample_batch(self, batch_size):
        idx = np.random.choice(self.n, batch_size, replace=False)
        return self.__getitem__(idx)

# ---------- tiny frozen MLP to nonlinearly map latents -> label logits ----------
def _make_mlp(in_dim, hidden_dim=100, out_dim=1, layers=2):
    blocks, d = [], in_dim
    for _ in range(layers):
        blocks += [nn.Linear(d, hidden_dim), nn.ReLU()]
        d = hidden_dim
    blocks += [nn.Linear(d, out_dim)]
    mlp = nn.Sequential(*blocks)
    for p in mlp.parameters():
        p.requires_grad = False
    return mlp

def _normalize(C, eps=1e-6):
    C = C - C.mean(axis=0, keepdims=True)
    s = C.std(axis=0, keepdims=True)
    s = np.where(s < eps, eps, s)
    return C / s

def _mix(A, B, frac_shared, normalize=True):
    """Return (1-frac_shared)*A + frac_shared*B with per-feature normalization so the fraction means 'information fraction'."""
    if normalize:
        A, B = _normalize(A), _normalize(B)
    return (1.0 - frac_shared) * A + frac_shared * B

# ---------- SIMPLE GENERATOR: one knob shared_frac controls X, Y, and label ----------
def generate_data_simple(
    n_samples,
    dim_info,               # {'Zs':ds,'Z1':d1,'Z2':d2,'X':dx,'Y':dy}
    shared_frac=0.5,        # single knob in [0,1]: 0=private-only, 1=shared-only (applies to X, Y, and label)
    noise_std=0.10,         # add Gaussian noise to X/Y
    seed=0,
    normalize_components=True,
    return_latents=True,
    hidden_dim=100
):
    """
    Returns:
        total_data: (2, N, D) if dx==dy else [X (N,dx), Y (N,dy)]
        labels: (N,) binary
        extras: dict with internals (optional)
    """
    if not (0.0 <= shared_frac <= 1.0):
        raise ValueError("shared_frac must be in [0,1].")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    ds, d1, d2 = dim_info['Zs'], dim_info['Z1'], dim_info['Z2']
    dx, dy     = dim_info['X'],  dim_info['Y']

    # --- sample true latents ---
    Zs = rng.normal(0, np.sqrt(0.5), size=(n_samples, ds)).astype(np.float32)
    Z1 = rng.normal(0, np.sqrt(0.5), size=(n_samples, d1)).astype(np.float32)
    Z2 = rng.normal(0, np.sqrt(0.5), size=(n_samples, d2)).astype(np.float32)

    # --- linear maps from latents to views ---
    T1p = rng.uniform(-1, 1, size=(d1, dx)).astype(np.float32)  # Z1 -> X
    T1s = rng.uniform(-1, 1, size=(ds, dx)).astype(np.float32)  # Zs -> X
    T2p = rng.uniform(-1, 1, size=(d2, dy)).astype(np.float32)  # Z2 -> Y
    T2s = rng.uniform(-1, 1, size=(ds, dy)).astype(np.float32)  # Zs -> Y

    X_priv, X_shared = Z1 @ T1p, Zs @ T1s
    Y_priv, Y_shared = Z2 @ T2p, Zs @ T2s

    # --- apply the SAME fraction to both modalities ---
    X = _mix(X_priv, X_shared, shared_frac, normalize=normalize_components)
    Y = _mix(Y_priv, Y_shared, shared_frac, normalize=normalize_components)

    if noise_std and noise_std > 0:
        X = X + rng.normal(0, noise_std, size=X.shape).astype(np.float32)
        Y = Y + rng.normal(0, noise_std, size=Y.shape).astype(np.float32)

    # --- ONE label with the SAME fraction: shared_frac shared + (1-shared_frac) private ---
    # Split the private portion evenly between Z1 and Z2 so it's still a single knob.
    w_sh = shared_frac
    w_p  = 1.0 - shared_frac
    w1 = w_p * 0.5
    w2 = w_p * 0.5

    # Build label input by concatenating weighted (normalized) latents, then a frozen MLP -> sigmoid -> median threshold.
    parts = []
    if d1 > 0 and w1 > 0: parts.append(_normalize(Z1) * w1)
    if ds > 0 and w_sh > 0: parts.append(_normalize(Zs) * w_sh)
    if d2 > 0 and w2 > 0: parts.append(_normalize(Z2) * w2)
    label_in = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    label_mlp = _make_mlp(label_in.shape[1], hidden_dim=hidden_dim, out_dim=1, layers=2)
    with torch.no_grad():
        raw_logits = label_mlp(torch.tensor(label_in, dtype=torch.float32)).squeeze(1).numpy()

    label_scale = 4.0          # try 4–6 for an easy regime
    label_noise_std = 0.00     # start at 0.00, then 0.02–0.05 later

    logits = label_scale * raw_logits
    if label_noise_std > 0.0:
        logits += rng.normal(0, label_noise_std, size=logits.shape).astype(np.float32)

    probs  = 1.0 / (1.0 + np.exp(-logits))
    thresh = np.median(probs)  # keeps classes roughly balanced
    labels = (probs >= thresh).astype(np.float32)

    # --- package ---
    if dx == dy:
        total_data = np.stack([X.astype(np.float32), Y.astype(np.float32)], axis=0)  # (2, N, D)
    else:
        total_data = [X.astype(np.float32), Y.astype(np.float32)]  # Dataset supports list too

    extras = None
    if return_latents:
        extras = dict(
            Zs=Zs, Z1=Z1, Z2=Z2, X=X, Y=Y,
            T1p=T1p, T1s=T1s, T2p=T2p, T2s=T2s,
            shared_frac=shared_frac, noise_std=noise_std
        )
    return total_data, labels, extras



class MultiViewDataset(Dataset):
    """
    Returns per item: a single list of length V+1
      sample = [x_view0, x_view1, ..., x_view{V-1}, y]
    Where each x_view{v} is a 1D np.float32 array of shape (S_v,)
    and y is an int64 label.

    With a default PyTorch DataLoader (no custom collate_fn), a batch becomes:
      batch[v] -> torch.FloatTensor of shape (B, S_v) for v in [0..V-1]
      batch[-1] -> torch.LongTensor of shape (B,)
    """
    def __init__(self, data_name, data_X, data_Y, norm_min=0):
        super(MultiViewDataset, self).__init__()
        self.data_name = data_name

        # Store views as a list for deterministic order [0..V-1]
        self.num_views = data_X.shape[0]
        self.X = []
        for v in range(self.num_views):
            self.X.append(self.normalize(data_X[v], min=norm_min))

        # Labels -> [0..C-1], int64
        self.Y = data_Y
        self.Y = np.squeeze(self.Y)
        if np.min(self.Y) == 1:
            self.Y = self.Y - 1
        self.Y = self.Y.astype(dtype=np.int64)
        self.num_classes = len(np.unique(self.Y))

        # Record dims as array of per-view feature sizes
        self.dims = self.get_dims()

    def __getitem__(self, index):
        # Build list of views for this sample
        sample = [self.X[v][index].astype(np.float32) for v in range(self.num_views)]
        # Append label as last item
        sample.append(self.Y[index])
        return sample  # length V+1

    def __len__(self):
        return len(self.X[0])

    def get_dims(self):
        # Return array shape (V, 1) with each view's feature size
        dims = []
        for v in range(self.num_views):
            dims.append([self.X[v].shape[1]])
        return np.array(dims)

    @staticmethod
    def normalize(x, min=0):
        # x: (N, D)
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    # -------------------------
    # Post-processing utilities
    # -------------------------
    def postprocessing(self, index, addNoise=False, sigma=0, ratio_noise=0.5, addConflict=False, ratio_conflict=0.5):
        """
        index: array-like of indices to consider (e.g., np.arange(len(dataset)))
        """
        if addNoise:
            self.addNoise(index, ratio_noise, sigma=sigma)
        if addConflict:
            self.addConflict(index, ratio_conflict)

    def addNoise(self, index, ratio, sigma):
        # Randomly choose subset of indices
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            # Random subset of views (possibly empty if randint(0) — ensure at least 1)
            k = np.random.randint(1, self.num_views + 1)
            views = np.random.choice(np.arange(self.num_views), size=k, replace=False)
            for v in views:
                # Add Gaussian noise elementwise to the feature vector
                self.X[v][i] = np.random.normal(self.X[v][i], sigma)

    def addConflict(self, index, ratio):
        # Pick one representative sample per class to swap-in as conflicting evidence
        records = dict()
        for c in range(self.num_classes):
            # Find one example of class c
            i_candidates = np.where(self.Y == c)[0]
            if len(i_candidates) == 0:
                continue
            i = i_candidates[0]
            # Record per-view vectors for this class prototype
            temp = dict()
            for v in range(self.num_views):
                temp[v] = self.X[v][i].copy()
            records[c] = temp

        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            v = np.random.randint(self.num_views)
            if len(records) == 0:
                continue
            # Replace one view with a prototype from a *different* class (cyclic next)
            self.X[v][i] = records[(self.Y[i] + 1) % self.num_classes][v]
        # Note: labels themselves are NOT changed — only feature/view conflict is injected.




def HandWritten():
    # dims of views: 240 76 216 47 64 6
    data_path = "data/handwritten.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['Y']
    return MultiViewDataset("HandWritten", data_X, data_Y)


def Scene():
    # dims of views: 20 59 40
    data_path = "data/scene15_mtv.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['gt']
    for v in range(len(data_X)):
        data_X[v] = data_X[v].T
    return MultiViewDataset("Scene", data_X, data_Y)


def PIE():
    # dims of views: 484 256 279
    data_path = "data/PIE_face_10.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['gt']
    for v in range(len(data_X)):
        data_X[v] = data_X[v].T
    return MultiViewDataset("PIE", data_X, data_Y)


def Caltech():
    # dims of views: 48 40 254 1984 512  928
    data_path = "data/Caltech101-20.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'].squeeze()
    data_Y = data['Y']
    return MultiViewDataset("Caltech", data_X, data_Y)


def CUB():
    # dims of views: 484 256 279
    data_path = "data/cub_googlenet_doc2vec_c10.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['gt']
    data_Y = data_Y - 1
    # for v in range(len(data_X)):
    #     data_X[v] = data_X[v].T
    return MultiViewDataset("CUB", data_X, data_Y)