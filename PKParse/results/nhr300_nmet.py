#2020
#f3d86a6de7f8cf4262f3b272206e26a9275cd1d8
import glob, math, time, datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from egnn_pytorch import EGNN_Network
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import os
os.environ["WANDB_MODE"] = "offline"
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
print("nhr3.py 25 June. biased/unshuffled (1.... 1... smallest dataset (that which made it through prepper). lets see th effects of data. ")

# 1) load entire dataset into RAM as torch.Tensors
class InMemoryNpzDataset(Dataset):
    def __init__(self, paths, pin_memory=False):
        self.data = []
        for p in paths:
            #try:
            a = np.load(p, allow_pickle=True)

            z= a["z"]
            #print(z)
            #z#=[zz for zz in z]
            #z=[zz for zz in z]
            #z=[[zzz for zzz in zz] for zz in z]
            #zs=[]
            #zzs=[]
            
            try:
                zs = [torch.tensor(z_i, dtype=torch.int32)   for z_i in z]
                xs = [torch.tensor(x_i, dtype=torch.float32) for x_i in a["pos"]]
                ys = [torch.tensor(y_i, dtype=torch.float32) for y_i in a["pks"]]
            except:
                print(p)
                continue
            if pin_memory:
                zs = [z.pin_memory() for z in zs]
                xs = [x.pin_memory() for x in xs]
                ys = [y.pin_memory() for y in ys]
            self.data.append((zs, xs, ys))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
np.random.seed(0)
to=time.time()
#files  = 100
ps=glob.glob("./pkegnn_inputs_nometal/*.npz")
np.random.shuffle(ps)
paths  = ps[0:100] #+ glob.glob("./inputs2/*.npz")[300:500] + glob.glob("./inputs2/*.npz")[1500:2700]# + glob.glob("./inputs2/*.npz")[120:]
tpaths = ps[100:140] #+ glob.glob("./inputs2/*.npz")[1300:1500] #+ glob.glob("./inputs2/*.npz")[2700:2800] 
batch_size=5

train_ds = InMemoryNpzDataset(paths)
val_ds   = InMemoryNpzDataset(tpaths)
print("NPZ in",len(train_ds), len(val_ds), (time.time() - to) / 60, "min")
def collate_graphs(batch):
    return batch

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=0, pin_memory=True, collate_fn=collate_graphs
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False,
    num_workers=0, pin_memory=True, collate_fn=collate_graphs
)
print("pinned and reaady",len(train_loader), len(val_loader), (time.time() - to) / 60, "min")
# 2) model pieces

# --- EGNN + FFN + residual block ---
# --- EGNN + FFN + residual block ---
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000, dropout=0.03):
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
    """todo: try to take head out here
    egnn_net --> layer norm --> ffn head"""
    def __init__(self, dim, depth,hidden_dim,dropout,
                 num_positions, num_tokens,
                 num_nearest_neighbors,
                 norm_coors,
                 num_global_tokens, num_edge_tokens):
        super().__init__()
        self.egnn = EGNN_Network(
            dim=dim, depth=depth, dropout=dropout,
            num_positions=num_positions,
            num_tokens=num_tokens,
            num_nearest_neighbors=num_nearest_neighbors,
            norm_coors=norm_coors,
            num_edge_tokens=num_edge_tokens,
            num_global_tokens=num_global_tokens,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim*dim),
            nn.PReLU(),# LU(),
            nn.Linear(hidden_dim*dim, dim),
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
    """TODO understand depth"""
    def __init__(self, dim, depth, hidden_dim, dropout,
                 num_positions, num_tokens,
                 num_nearest_neighbors,
                 norm_coors,
                 num_global_tokens, num_edge_tokens):
        super().__init__()
        self.blocks = nn.ModuleList([
            EGNNBlock(dim, depth, hidden_dim, dropout,
                      num_positions, num_tokens,
                      num_nearest_neighbors,
                      norm_coors,
                      num_global_tokens, num_edge_tokens)
            for _ in range(1)
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
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * hidden_dim),
            nn.PReLU(),
            nn.Linear(embed_dim * hidden_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    def forward(self, x, key_padding_mask=None):
        # x: [seq_len, batch, embed_dim]
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x    = self.norm1(x + a)
        f    = self.ffn(x)
        x    = self.norm2(x + f)
        return x,_

# --- RBF with learnable cutoff ---
class LearnableRBF(nn.Module):
    """TODO change cutout"""
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
    def __init__(self, in_channels,hidden_dim):
        super().__init__()
        #self.conv5    = nn.Conv1d(1, 1, 7, padding=3) #operates on seq
        self.lin = nn.Linear(in_channels,1)
        self.lin2 =nn.Linear(1,1)
        
        #pk
        self.scalar_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        2
        return #self.convb(x.permute(2,1,0)) 

# --- builder for the stacked EGNN ---
def build_egnn(dim,depth,hidden_dim,num_neighbors, num_edge_tokens,num_global_tokens,dropout):
    return StackedEGNN(
        dim=dim, depth=depth, hidden_dim=hidden_dim,
        dropout=dropout,
        num_positions=1000, num_tokens=78,
        num_nearest_neighbors=num_neighbors,
        norm_coors=True,
        num_edge_tokens=num_edge_tokens,
        num_global_tokens=num_global_tokens
    )

# 3) instantiate everything
dim, basis = 10, 64 #scale to 3,16 at least # dim must be divisible by 2
depth=2 #scale to 2, at least
hidden_dim=3
num_heads=dim + basis 
num_edge_tokens=256
num_global_tokens=256
dropout=0.03 # here
cutoff=20.0
epochs=150
num_neighbors=2
net   = build_egnn(dim,depth,hidden_dim,num_neighbors,num_edge_tokens,num_global_tokens,dropout).to(device)
A     = PositionalEncoding(dim+basis).to(device)
mha   = AttentionBlock(embed_dim=dim+basis, num_heads=num_heads, hidden_dim=hidden_dim).to(device)
RBF   = LearnableRBF(num_basis=basis, cutoff=cutoff).to(device) 
model = TinyRegressor(in_channels=basis+dim,hidden_dim=dim+basis).to(device)
runid=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

#8)

criterion = nn.L1Loss()
optimizer = torch.optim.AdamW([
    {"params": net.parameters(),   "lr":5e-3},
    {"params": mha.parameters(),   "lr":5e-3},
    {"params": model.parameters(), "lr":5e-3},
    {"params": RBF.parameters(),   "lr":5e-3}])
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1)
print("scheduler 5e-3. num nearest neiighbors 2 cutoff 9 num heads = in_dim.")#
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=.99, patience=0, cooldown=0, min_lr=1e-8, verbose=False)
# 4) training + validation
train_hist, val_hist = [], []
print("runid:",runid)
print("dim = num heads embeded dim:", dim)
print("depth",depth)
print("hidden_dim:",hidden_dim)
print("basis", basis)
print(f"hidden dim 8. lower optimizer and inc batch size to 25 from 5. changed mha to act on all and more thx to mam and chat anon and more")#rbf cut 4 same as real cutouts, only .01 dropout regularization, batch size 250. full val, {len(train_loader)} train")
print("")
#print(optimizer.state_dict())
print("FIX SCHEDULER Max schedular cool 0 wait 0 min 1e-8 patience zero cooldown zero")
print(criterion)
"""out1 N, dim
attn out N, N, 2
pairwise Ds 1,1,N
rbf 1 1 N n_rbf
rbf N basis
model input N 3 3
gnode N 1
gemb 1 nasis+1 n 
pooled 1 1 1
"""
# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="Biomodeling",
    # Set the wandb project where this run will be logged.
    project="Overfit PKEGNN on num_atoms = 500",
    # Track hyperparameters and run metadata.
    config={"runid": runid,
        "learning_rate": [op["lr"] for op in optimizer.param_groups], #net mha model rbf
        "dataset": f"inputs3 (num neighbors = 500) {len(paths) + len(tpaths)}",
        "batch size": batch_size,
        "epochs": epochs,
        "dim": dim,
        "depth": depth,
        "basis": basis,
        "num edge and global tokens": [num_edge_tokens,num_global_tokens],
        "dropout": [dropout, 0.03], #egnn p.enc. 
        "rbf cutoff": cutoff,
        "scheduler": [scheduler.min_lrs,scheduler.cooldown,scheduler.factor,"updates per epoch"],
        "loss": criterion,
        "architecture": b"EGNN (1 blocks) --> coords thru rbf --> concat rbf_out + egnn_out[0] ==> MHA --> model.linear --> pool > all protein's outputs thru model.linear2"})
vl,tl=[],[]
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    net.train(); mha.train(); model.train(); RBF.train()
    epoch_train_losses = []

    # training
    for batch in train_loader:
        optimizer.zero_grad()
        for zs, xs, ys in batch:
            outs = []
            for z_t, x_t in zip(zs, xs):
                z_t = z_t.to(device).unsqueeze(0)
                x_t = x_t.to(device).unsqueeze(0)
                with autocast():
                    out1, coords = net(z_t, x_t)

                    #dmat = pairwise_distances(coords)
                    rbf  = RBF(pairwise_distances(coords))
                    b=torch.concat((rbf[:,0].T,out1[0].T.unsqueeze(2)))
                    en = A(b.permute(1,2,0)) #encoding
                    c=model.lin(mha(en)[0])

                    pooled = model.scalar_pool(c)[0][0] 

                outs.append(pooled)
            
            preds  = torch.stack(outs).to(device)
            #print(preds.shape,"preds")
            
            target = torch.hstack(ys).to(device).flatten()
            with autocast():
                #preds2=model.conv5(preds.unsqueeze(1)).to(device).flatten()
                pred3=model.lin2(preds).flatten().to(device)
            #pred3=model.relu(preds)
            #pred3=model.rel(preds)
            loss = criterion(pred3, target)
            #print(loss)
            #print(pred3)
            if use_amp:
                #with autocast():
                scaler.scale(loss).backward()
                #scheduler.step(loss)
                
            else:
                loss.backward()
            epoch_train_losses.append(loss.item())

        if use_amp:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()

    #train_hist.append(np.mean(epoch_train_losses))
    

    # validation
    print(loss)
    net.eval(); mha.eval(); model.eval(); RBF.eval()
    epoch_val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            for zs, xs, ys in batch:
                outs = []
                for z_t, x_t in zip(zs, xs):
                    z_t = z_t.to(device).unsqueeze(0)
                    x_t = x_t.to(device).unsqueeze(0)
                    with autocast():
                        out1, coords = net(z_t, x_t)
                        rbf  = RBF(pairwise_distances(coords))
                        b=torch.concat((rbf[:,0].T,out1[0].T.unsqueeze(2)))
                        en = A(b.permute(1,2,0)) #encoding
                        c=model.lin(mha(en)[0])

                    pooled = model.scalar_pool(c)[0][0] 
                    outs.append(pooled)
                
                preds  = torch.stack(outs).to(device)
                #print(preds.shape,"preds")
                
                target = torch.hstack(ys).to(device).flatten()
                
                with autocast():
                    #preds2=model.conv5(preds.unsqueeze(1)).to(device).flatten()
                    pred3=model.lin2(preds).flatten().to(device)
                loss = criterion(pred3, target)
                epoch_val_losses.append(loss)
                #loss = torch.mean(epoch_val_losses)
                #scheduler.step(loss)  # val_loss = your average validation loss
                #scheduler.step(loss) 
            #e
            #print(val_loss.item())

                    # 5) save a single timestamped checkpoint
    
    loss=torch.mean(torch.tensor(epoch_val_losses))
    scheduler.step(loss) 
    tl.append(epoch_train_losses)
    vl.append(epoch_val_losses)
    elapsed_min = (time.time() - t0) / 60
    


    #print(torch.mean(torch.tensor(epoch_train_losses)),loss)
    
# 5) save a single timestamped checkpoint
    elapsed_min = (time.time() - t0) / 60
    print("pooled",pooled, pred3[-1],target[-1])
    print(elapsed_min,"min")
                #print(val_loss.item())

                    # 5) save a single timestamped checkpoint
    
    #loss=torch.mean(torch.tensor(epoch_val_losses))
    #scheduler.step(loss) 
    vloss,tloss=loss.item(),torch.mean(torch.tensor(epoch_train_losses)).item()
    #log
    run.log({"tloss": tloss, "vloss": vloss, "elapsed_time": elapsed_min})
    #elapsed_min = (time.time() - t0) / 60
    #print(len(epoch_train_losses),len(epoch_val_losses))
    #timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint = {
        "epoch":         epoch+1,
        "elapsed_min":   elapsed_min,
    #    "net":           net.state_dict(),
    #    "mha":           mha.state_dict(),
    #    "model":         model.state_dict(),
    #    "rbf":           RBF.state_dict(),
    #    "optimizer":     optimizer.state_dict(),
        "train_history": tl,
        "val_history":   vl}
    #}
    torch.save(checkpoint, f"./{runid}-checkpoint.pt")
    #print(f"Saved checkpoint_{timestamp}.pt ({elapsed_min:.1f} min)")

    #val_hist.append(loss.item())
    print(f" → avg train loss: {tloss:.4f}")
    print(f" → avg   val loss: {vloss:.4f}")
run.finish()

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
    "scheduler":     scheduler.state_dict(),
    "train_history": tl,
    "val_history":   vl,
}
torch.save(checkpoint, f"./{runid}-checkpoint_{timestamp}.pt")
print(f"Saved checkpoint_{timestamp}.pt ({elapsed_min:.1f} min)")
#os.system("wandb sync --include-offline --sync-all wandb")



