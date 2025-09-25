import paths
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, Linear, HeteroConv
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
import numpy as np

# load data
transactions = pd.read_parquet(paths.DATA_PATH / "transactions.parquet")
counterparties = pd.read_parquet(paths.DATA_PATH / "counterparties.parquet")
invoices = pd.read_parquet(paths.DATA_PATH / "invoices.parquet")
users = pd.read_parquet(paths.DATA_PATH / "users.parquet")

# index mapping
tx_map = {tx_id: i for i, tx_id in enumerate(transactions["tx_id"])}
cp_map = {cp_id: i for i, cp_id in enumerate(counterparties["cp_id"])}
inv_map = {inv_id: i for i, inv_id in enumerate(invoices["inv_id"])}
usr_map = {user_id: i for i, user_id in enumerate(users["user_id"])}

data = HeteroData()


# add node features
def zscore(df, cols):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols].fillna(0.0).astype(float))
    return torch.tensor(X, dtype=torch.float)


tx_feats = zscore(
    transactions.assign(
        amount_abs=transactions["amount"].abs(),
        memo_len=transactions["memo"].fillna(0.0).str.len(),
        month=pd.to_datetime(transactions["date"]).dt.month,
        dow=pd.to_datetime(transactions["date"]).dt.dayofweek,
    ),
    cols=["amount_abs", "memo_len", "month", "dow"],
)
data["transaction"].x = tx_feats

cp_feats = pd.get_dummies(counterparties[["country"]], dummy_na=True)
data["counterparty"].x = torch.tensor(cp_feats.values, dtype=torch.float)

inv_feats = zscore(
    invoices.assign(
        age_days=(
            pd.to_datetime(invoices.issue_date) - pd.to_datetime(invoices.due_date)
        ).dt.days.fillna(0)
    ),
    ["vat_amount", "age_days"],
)
data["invoice"].x = inv_feats

usr_feats = pd.get_dummies(users[["dept", "role"]], dummy_na=True)
data["user"].x = torch.tensor(usr_feats.values, dtype=torch.float)


# Edges
src = [tx_map[r.tx_id] for _, r in transactions.iterrows()]
dst = [usr_map[r.user_id] for _, r in transactions.iterrows()]
data["transaction", "posted_by", "user"].edge_index = torch.tensor(
    [src, dst], dtype=torch.long
)

src = [tx_map[r.tx_id] for _, r in transactions.iterrows()]
dst = [cp_map[r.counterparty_id] for _, r in transactions.iterrows()]
data["transaction", "paid_to", "counterparty"].edge_index = torch.tensor(
    [src, dst], dtype=torch.long
)

src = [tx_map[r.tx_id] for _, r in transactions.iterrows()]
dst = [inv_map[r.invoice_id] for _, r in transactions.iterrows()]
data["transaction", "references", "invoice"].edge_index = torch.tensor(
    [src, dst], dtype=torch.long
)

# Add self-loops for 'transaction' nodes
num_tx_nodes = data["transaction"].x.size(0)
self_loop_src = torch.arange(num_tx_nodes)
self_loop_dst = torch.arange(num_tx_nodes)

# Add self-loops to the edge index
if ("transaction", "self_loop", "transaction") not in data.edge_types:
    data["transaction", "self_loop", "transaction"].edge_index = torch.stack(
        [self_loop_src, self_loop_dst], dim=0
    )

# Counterparty similarity edges (same IBAN)
iban_to_cp = counterparties.groupby("iban")["cp_id"].apply(list)
pair_src, pair_dst = [], []
for ib, cplist in iban_to_cp.items():
    idxs = [cp_map[c] for c in cplist if c in cp_map]

    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            pair_src += [idxs[i], idxs[j]]
            pair_dst += [idxs[j], idxs[i]]

if pair_src:
    data["counterparty", "same_bank_iban", "counterparty"].edge_index = torch.tensor(
        [pair_src, pair_dst]
    )

# Labels and masks
if "is_risky" in transactions.columns:
    y = torch.tensor(
        transactions["is_risky"].fillna(0).astype(int).values, dtype=torch.long
    )
    data["transaction"].y = y
    n = y.size(0)
    idx = np.arange(n)
    np.random.shuffle(idx)
    tr, va = int(0.7 * n), int(0.85 * n)

    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[idx[:tr]] = True

    val_mask = torch.zeros(n, dtype=torch.bool)
    val_mask[idx[tr:va]] = True

    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[idx[va:]] = True

    data["transaction"].train_mask = train_mask
    data["transaction"].val_mask = val_mask
    data["transaction"].test_mask = test_mask


# RGCN Model
# RGCN Model
class HeteroRGCN(nn.Module):
    def __init__(self, metadata, hidden=64, out=2):
        super().__init__()
        self.lin_in = nn.ModuleDict(
            {ntype: Linear(-1, hidden) for ntype in metadata[0]}
        )
        self.conv1 = HeteroConv(
            {edge_type: SAGEConv((-1, -1), hidden) for edge_type in metadata[1]},
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {edge_type: SAGEConv((-1, -1), hidden) for edge_type in metadata[1]},
            aggr="sum",
        )
        self.tx_head = nn.Sequential(
            Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2), Linear(hidden, out)
        )

    def forward(self, x_dict, edge_index_dict):
        # Apply initial linear transformation
        x_dict = {k: self.lin_in[k](v) for k, v in x_dict.items()}

        # Apply the first HeteroConv layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: torch.relu(v) for k, v in x_dict.items()}

        # Apply the second HeteroConv layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: torch.relu(v) for k, v in x_dict.items()}

        # Compute logits for the "transaction" node type
        logits = self.tx_head(x_dict["transaction"])

        return logits, x_dict


metadata = data.metadata()
model = HeteroRGCN(metadata, hidden=64, out=2)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()

# train and evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)
model = model.to(device)


def step(mask, train=True):
    model.train(train)
    logits, _ = model(data.x_dict, data.edge_index_dict)
    loss = crit(logits[mask], data["transaction"].y[mask])
    if train:
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = logits.softmax(dim=-1).argmax(-1)
        acc = (pred[mask] == data["transaction"].y[mask]).float().mean().item()
    return loss.item(), acc


for epoch in range(1, 100):
    tl, ta = step(data["transaction"].train_mask, train=True)
    vl, va = step(data["transaction"].val_mask, train=False)

    if epoch % 10 == 0:
        print(
            f"epoch {epoch}  train_loss {tl:.3f} acc {ta:.3f}  val_loss {vl:.3f} acc {va:.3f}"
        )

# Test
model.eval()
with torch.no_grad():
    logits, emb = model(data.x_dict, data.edge_index_dict)
    test_mask = data["transaction"].test_mask
    pred = logits.softmax(dim=-1).argmax(-1)
    test_acc = (
        (pred[test_mask] == data["transaction"].y[test_mask]).float().mean().item()
    )
    risk_score = logits.softmax(dim=-1)[:, 1].detach().cpu().numpy()

print(f"test_acc {test_acc:.3f}")
