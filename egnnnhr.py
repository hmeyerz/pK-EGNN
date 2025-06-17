

import glob, math, time, datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from egnn_pytorch import EGNN_Network

# 0) start timer
t0 = time.time()

# reproducibility + device
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# decide AMP only on GPU
use_amp = (device.type == "cuda")
if use_amp:
    scaler = GradScaler()
else:
    class DummyCM:
        def __enter__(self): pass
        def __exit__(self, *args): pass
    autocast = DummyCM
    scaler   = None

print(f"Running on {device}; mixed-precision = {use_amp}")

# 1) load entire dataset into RAM as torch.Tensors
class InMemoryNpzDataset(Dataset):
    def __init__(self, paths, pin_memory=False):
        self.data = []
        for p in paths:
            a = np.load(p, allow_pickle=True)
            zs = [torch.tensor(z_i, dtype=torch.int32)   for z_i in a["z"]]
            xs = [torch.tensor(x_i, dtype=torch.float32) for x_i in a["pos"]]
            ys = [torch.tensor(y_i, dtype=torch.float32) for y_i in a["pks"]]
            if pin_memory:
                zs = [z.pin_memory() for z in zs]
                xs = [x.pin_memory() for x in xs]
                ys = [y.pin_memory() for y in ys]
            self.data.append((zs, xs, ys))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


#files  = 100
paths  = glob.glob("./inputs2/*.npz")[200:]# + glob.glob("./inputs2/*.npz")[120:]
tpaths = glob.glob("./inputs2/*.npz")[100:200]

train_ds = InMemoryNpzDataset(paths)
val_ds   = InMemoryNpzDataset(tpaths)

def collate_graphs(batch):
    return batch

train_loader = DataLoader(
    train_ds, batch_size=50, shuffle=True,
    num_workers=0, pin_memory=True, collate_fn=collate_graphs
)
val_loader = DataLoader(
    val_ds, batch_size=50, shuffle=False,
    num_workers=0, pin_memory=True, collate_fn=collate_graphs
)

# 2) model pieces

# --- EGNN + FFN + residual block ---
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.01, max_len=2000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float)
                        * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        cosp = torch.cos(pos * div)
        pe[:, 1::2] = cosp[:, : pe[:, 1::2].shape[1]]
        self.register_buffer('pe', pe.unsqueeze(1))
    def forward(self, x):
        return self.dropout(x + self.pe[: x.size(0)])

class EGNNBlock(nn.Module):
    def __init__(self, dim, depth,
                 num_positions, num_tokens,
                 num_nearest_neighbors,
                 norm_coors,
                 num_global_tokens, num_edge_tokens):
        super().__init__()
        self.egnn = EGNN_Network(
            dim=dim, depth=depth,
            num_positions=num_positions,
            num_tokens=num_tokens,
            num_nearest_neighbors=num_nearest_neighbors,
            norm_coors=norm_coors,
            num_global_tokens=num_global_tokens,
            num_edge_tokens=num_edge_tokens,
            dropout=0.03
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, z, x):
        (h_list, coords) = self.egnn(z, x)
        h = h_list[0]  # [B,N,dim]
        h2 = h
        h  = self.norm1(h + h2)
        h2 = self.ffn(h)
        h  = self.norm2(h + h2)
        return [h], coords

# --- stack multiple EGNNBlocks ---
class StackedEGNN(nn.Module):
    def __init__(self, dim, depth,
                 num_positions, num_tokens,
                 num_nearest_neighbors,
                 norm_coors,
                 num_global_tokens, num_edge_tokens):
        super().__init__()
        self.blocks = nn.ModuleList([
            EGNNBlock(dim, depth,
                      num_positions, num_tokens,
                      num_nearest_neighbors,
                      norm_coors,
                      num_global_tokens, num_edge_tokens)
            for _ in range(depth)
        ])
    def forward(self, z, x):
        coords = x
        h_list = None
        for block in self.blocks:
            if h_list is None:
                h_list, coords = block(z, x)
            else:
                h_list, coords = block(z, coords)
        return h_list, coords

# --- Transformer‐style AttentionBlock ---
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_mult=4):
        super().__init__()
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_mult),
            nn.GELU(),
            nn.Linear(embed_dim * ffn_mult, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    def forward(self, x, key_padding_mask=None):
        # x: [seq_len, batch, embed_dim]
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x    = self.norm1(x + a)
        f    = self.ffn(x)
        x    = self.norm2(x + f)
        return x

# --- RBF with learnable cutoff ---
class LearnableRBF(nn.Module):
    def __init__(self, num_basis=16, cutoff=5.0):
        super().__init__()
        self.cutoff = nn.Parameter(torch.tensor(cutoff))
        self.mu     = nn.Parameter(torch.linspace(0.0, 1.0, num_basis))
        self.gamma  = nn.Parameter(torch.tensor(12.0))
    def forward(self, dist):
        # dist: [B,N,N]
        mu = self.mu * self.cutoff     # [K]
        d  = dist.unsqueeze(-1)        # [B,N,N,1]
        return torch.exp(-self.gamma * (d - mu)**2)

def pairwise_distances(x):
    return torch.norm(x.unsqueeze(1) - x.unsqueeze(0), dim=-1)

def aggregate_rbf_features(rbf):
    return rbf.mean(dim=(0,1))

# --- TinyRegressor unchanged ---
class TinyRegressor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        #self.relu     = nn.PReLU()
        self.conv1    = nn.Conv1d(in_channels, 2, 2)
        self.conv4    = nn.Conv1d(2, 1, 1)
        self.conv5    = nn.Conv1d(1, 1, 7, padding=3)
        self.poolmha  = nn.AdaptiveMaxPool2d(3)
        self.poolmha2 = nn.AdaptiveMaxPool2d(1)
        self.pool     = nn.AdaptiveMaxPool1d(1)
    def forward(self, x):
        #x = self.relu(x)
        x = self.conv1(x)
        #x = self.relu(x)
        return self.conv4(x)

# --- builder for the stacked EGNN ---
def build_egnn(dim,depth):
    return StackedEGNN(
        dim=dim, depth=depth,
        num_positions=500, num_tokens=78,
        num_nearest_neighbors=2,
        norm_coors=True,
        num_global_tokens=256,
        num_edge_tokens=256
    )

# 3) instantiate everything
dim, basis = 2, 16
depth=2
net   = build_egnn(dim,depth).to(device)
A     = PositionalEncoding(dim).to(device)
mha   = AttentionBlock(embed_dim=dim, num_heads=dim).to(device)
RBF   = LearnableRBF(num_basis=basis, cutoff=5.0).to(device)
model = TinyRegressor(in_channels=basis+1).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.AdamW([
    {"params": net.parameters(),   "lr":1e-3},
    {"params": mha.parameters(),   "lr":1e-4},
    {"params": model.parameters(), "lr":1e-4},
    {"params": RBF.parameters(),   "lr":1e-4},
], weight_decay=1e-5)

# 4) training + validation
train_hist, val_hist = [], []
for epoch in range(30):
    print(f"Epoch {epoch}")
    net.train(); mha.train(); model.train(); RBF.train()
    epoch_train_losses = []

    # training
    for batch in train_loader:
        optimizer.zero_grad()
        for zs, xs, ys in batch:
            outs = []
            for z_t, x_t, y_t in zip(zs, xs, ys):
                z_t = z_t.to(device).unsqueeze(0)
                x_t = x_t.to(device).unsqueeze(0)
                with autocast():
                    out1, coords = net(z_t, x_t)
                    h = out1[0]                     # [1,N,dim]
                    attn_in = A(h).permute(1,0,2)   # [N,1,dim]
                    attn_out= mha(attn_in)          # [N,1,dim]
                    attn_out= attn_out.permute(1,0,2)  # [1,N,dim]

                    dmat = pairwise_distances(coords).to(device)
                    rbf  = RBF(dmat)
                    grbf = aggregate_rbf_features(rbf).to(device)

                    gnode= model.poolmha2(model.poolmha(attn_out))[:,0].to(device)
                    gemb = torch.hstack([gnode, grbf]).unsqueeze(2).to(device)
                    e_pred = model(gemb.permute(2,1,0))
                    pooled = model.pool(e_pred)
                outs.append(pooled)
            
            preds  = torch.hstack(outs).to(device).flatten()
            
            target = torch.hstack(ys).to(device).flatten()
            with autocast():
                preds=model.conv5(preds.unsqueeze(0)).to(device).flatten()
                loss = criterion(preds, target)
            if use_amp:
                
                scaler.scale(loss).backward()
            else:
                loss.backward()
            epoch_train_losses.append(loss.item())

        if use_amp:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()

    train_hist.append(epoch_train_losses)

    # validation
    net.eval(); mha.eval(); model.eval(); RBF.eval()
    epoch_val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            for zs, xs, ys in batch:
                outs = []
                for z_t, x_t, y_t in zip(zs, xs, ys):
                    z_t = z_t.to(device).unsqueeze(0)
                    x_t = x_t.to(device).unsqueeze(0)
                    out1, coords = net(z_t, x_t)
                    h = out1[0]
                    attn_in = A(h).permute(1,0,2)
                    attn_out= mha(attn_in)
                    attn_out= attn_out.permute(1,0,2)

                    dmat = pairwise_distances(coords).to(device)
                    rbf  = RBF(dmat)
                    grbf = aggregate_rbf_features(rbf).to(device)

                    gnode= model.poolmha2(model.poolmha(attn_out))[:,0].to(device)
                    gemb = torch.hstack([gnode, grbf]).unsqueeze(2).to(device)
                    e_pred = model(gemb.permute(2,1,0))
                    pooled = model.pool(e_pred)
                    outs.append(pooled)
                with autocast():
                    preds  = torch.hstack(outs).to(device).flatten()
                    preds=model.conv5(preds.unsqueeze(0)).to(device).flatten()
             
                #preds  = torch.hstack(outs).to(device).flatten()
                target = torch.hstack(ys).to(device).flatten()
                val_loss = criterion(preds, target)
                epoch_val_losses.append(val_loss.item())

                    # 5) save a single timestamped checkpoint
    elapsed_min = (time.time() - t0) / 60
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint = {
        "epoch":         epoch+1,
        "elapsed_min":   elapsed_min,
        "net":           net.state_dict(),
        "mha":           mha.state_dict(),
        "model":         model.state_dict(),
        "rbf":           RBF.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "train_history": train_hist,
        "val_history":   val_hist,
    }
    torch.save(checkpoint, f"./checkpoint_{timestamp}.pt")
    print(f"Saved checkpoint_{timestamp}.pt ({elapsed_min:.1f} min)")

    val_hist.append(epoch_val_losses)
    print(f" → avg train loss: {np.mean(epoch_train_losses):.4f}")
    print(f" → avg   val loss: {np.mean(epoch_val_losses):.4f}")

# 5) save a single timestamped checkpoint
elapsed_min = (time.time() - t0) / 60
timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint = {
    "epoch":         epoch+1,
    "elapsed_min":   elapsed_min,
    "net":           net.state_dict(),
    "mha":           mha.state_dict(),
    "model":         model.state_dict(),
    "rbf":           RBF.state_dict(),
    "optimizer":     optimizer.state_dict(),
    "train_history": train_hist,
    "val_history":   val_hist,
}
torch.save(checkpoint, f"./checkpoint_{timestamp}.pt")
print(f"Saved checkpoint_{timestamp}.pt ({elapsed_min:.1f} min)")
