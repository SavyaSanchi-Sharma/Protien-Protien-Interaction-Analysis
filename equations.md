### **RSA**

$$
\text{RSA}_i = \min\left(\max\left(\frac{\text{ASA}_i^{\text{DSSP}}}{\text{ASA}_{\max}(\text{aa}_i)},\ 0\right),\ 1\right)
$$

* $\text{RSA}_i$: Relative Solvent Accessibility of residue $i$ — variable `rsa` in [dataprep/dataprep.py:257](dataprep/dataprep.py#L257)
* $\text{ASA}_i^{\text{DSSP}}$: Accessible Surface Area from DSSP — `dssp[(chain_id, r.id)][3]` in [dataprep/dataprep.py:257](dataprep/dataprep.py#L257)
* $\text{ASA}_{\max}(\text{aa}_i)$: Maximum ASA for that amino acid type — `MAX_ASA` dict in [dataprep/dataprep.py:26-31](dataprep/dataprep.py#L26-L31)
* $\text{aa}_i$: Amino acid at position $i$ — `aa = seq1(r.resname)` in [dataprep/dataprep.py:254](dataprep/dataprep.py#L254)
* Clipping is `np.clip(..., 0.0, 1.0)`

---

### **Flexibility**

$$
\text{Flex}_i = \frac{B_i - \min_j B_j}{\max_j B_j - \min_j B_j}
$$

* $B_i$: B-factor (temperature factor) of residue $i$ — `r["CA"].get_bfactor()` collected in array `b` at [dataprep/dataprep.py:222](dataprep/dataprep.py#L222)
* $\min_j B_j$: Minimum B-factor in protein — `b.min()`
* $\max_j B_j$: Maximum B-factor in protein — `b.max()`
* Result stored in array `flex` at [dataprep/dataprep.py:224](dataprep/dataprep.py#L224); fallback to zeros if `b_range < 1e-6`

---

### **Hydrophobicity**

$$
\text{Hydro}_i = \frac{1}{6}\sum_{k=1}^{6} S_k(\text{aa}_i)
$$

* $\text{Hydro}_i$: Hydrophobicity of residue $i$ — variable `hydro` in [dataprep/dataprep.py:261](dataprep/dataprep.py#L261)
* $S_k$: Hydrophobicity scale $k$ — `HYDRO_SCALES[k]` in [dataprep/dataprep.py:33-40](dataprep/dataprep.py#L33-L40) (Kyte-Doolittle, Hopp-Woods, Eisenberg, Wimley-White, Hessa, Janin)
* $k$: Index over 6 scales — implicit in `np.mean([s.get(aa, 0.0) for s in HYDRO_SCALES])`

---

### **Packing Density**

$$
\rho_i = \frac{\left|\left\{j : \|x_j - x_i^{C\alpha}\| < 3.5\right\}\right|}{\left|\{\text{atoms of residue } i\}\right|}
$$

* $\rho_i$: Packing density — array `dens` in [dataprep/dataprep.py:226-228](dataprep/dataprep.py#L226-L228); min-max normalized at [dataprep/dataprep.py:231](dataprep/dataprep.py#L231)
* $x_i^{C\alpha}$: Position of C-alpha atom of residue $i$ — `r["CA"].coord`
* $x_j$: Position of atom $j$ — entries in `all_atoms` flattened across all residues, fed into `KDTree(np.array(all_atoms))` at [dataprep/dataprep.py:215-219](dataprep/dataprep.py#L215-L219)
* $\|\cdot\|$: Euclidean distance — implicit in `tree.query_ball_point(r["CA"].coord, 3.5)`
* The 3.5 Å radius is hard-coded at [dataprep/dataprep.py:227](dataprep/dataprep.py#L227)

---

### **Half-Sphere Exposure**

$$
\mathbf{r}_i = \mathbf{x}_i^{C\beta} - \mathbf{x}_i^{C\alpha}
$$
$$
\text{up}_i = \left|\left\{j : \mathbf{r}_i \cdot (\mathbf{x}_j^{C\alpha} - \mathbf{x}_i^{C\alpha}) \ge 0\right\}\right|
$$
$$
\text{down}_i = N_i - \text{up}_i - 1
$$

* $\mathbf{r}_i$: Reference vector (Cβ − Cα) — variable `ref` in [dataprep/dataprep.py:284](dataprep/dataprep.py#L284); falls back to `(N − Cα)` for glycine
* $\cdot$: Dot product — `np.dot(ca_coords[j] - ca, ref)` in [dataprep/dataprep.py:290](dataprep/dataprep.py#L290)
* $N_i$: Number of neighboring residues — `len(neigh)` from `ca_tree.query_ball_point(ca, 8.0)` in [dataprep/dataprep.py:285](dataprep/dataprep.py#L285)
* $\text{up}_i, \text{down}_i$: Counts in `up`, `down` at [dataprep/dataprep.py:286-293](dataprep/dataprep.py#L286-L293); reported as ratios `up / total`, `down / total` (`total = max(len(neigh) - 1, 1)`)
* The 8 Å neighborhood radius is hard-coded at [dataprep/dataprep.py:285](dataprep/dataprep.py#L285)

---

### **Polynomial Interaction**

$$
\Phi_i = \left[1,\ \text{RSA}_i,\ \text{Flex}_i,\ \text{RSA}_i \cdot \text{Flex}_i\right]
$$

* $\Phi_i$: Feature vector for residue $i$ — variable `poly` in [dataprep/dataprep.py:262](dataprep/dataprep.py#L262), four floats: `[1.0, rsa, float(flex[i]), rsa * float(flex[i])]`
* Stored in CSV columns `Poly_bias, Poly_RSA, Poly_Flex, Poly_interaction` (see [dataprep/dataprep.py:17-24](dataprep/dataprep.py#L17-L24))

---

### **Inter-Peptide Angle**

$$
\mathbf{v}_1 = \mathbf{c} - \mathbf{n},\quad \mathbf{v}_2 = \mathbf{o} - \mathbf{n}
$$
$$
\theta_i = \arccos\left(\frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\|\ \|\mathbf{v}_2\|}\right)
$$

* $\theta_i$: Bond angle — variable `bond` in [dataprep/dataprep.py:265-271](dataprep/dataprep.py#L265-L271), computed via `Vector.angle(...)` from Bio.PDB
* $\mathbf{c}$: C-atom of previous residue — `residues[i-1]["C"].get_vector()`
* $\mathbf{n}$: N-atom of current residue — `r["N"].get_vector()`
* $\mathbf{o}$: O-atom of next residue — `residues[i+1]["O"].get_vector()`
* For first/last residues `bond = 0.0`

---

### **Torsion Angle**

$$
\omega_i = \arccos\left(\frac{(\mathbf{N}-\mathbf{C\alpha}) \cdot (\mathbf{O}-\mathbf{C})}{\|\mathbf{N}-\mathbf{C\alpha}\|\ \|\mathbf{O}-\mathbf{C}\|}\right)\cdot \frac{180}{\pi}
$$

* $\omega_i$: Torsion angle — variable `omega` in [dataprep/dataprep.py:274-281](dataprep/dataprep.py#L274-L281); stored as `sin(omega), cos(omega)` in CSV columns `sin_omega, cos_omega`
* $\mathbf{N}, \mathbf{C\alpha}, \mathbf{C}, \mathbf{O}$: backbone atom positions — `r["N"].get_vector()`, `r["CA"].get_vector()`, `r["C"].get_vector()`, `r["O"].get_vector()`
* φ, ψ are extracted from DSSP (`dssp[k][4]`, `dssp[k][5]`) and cubic-spline interpolated for missing values at [dataprep/dataprep.py:233-248](dataprep/dataprep.py#L233-L248), stored as `sin_phi, cos_phi, sin_psi, cos_psi`

---

### **Adjacency Matrix**

$$
A_{ij} = \mathbb{1}\left[\|x_i^{C\alpha} - x_j^{C\alpha}\| \le 14\right]
$$

* $A_{ij}$: Edge between residues $i, j$ — built in [esm_model/createdatset.py:96-134](esm_model/createdatset.py#L96-L134) as `edge_index`
* $\mathbb{1}[\cdot]$: Indicator function — implemented via `tree.query_ball_point(coords[i], self.cutoff)` at [esm_model/createdatset.py:122](esm_model/createdatset.py#L122)
* The 14 Å cutoff is the constructor default `cutoff=14.0` at [esm_model/createdatset.py:33](esm_model/createdatset.py#L33)
* `edge_weight` is set to all-ones at [esm_model/createdatset.py:133](esm_model/createdatset.py#L133)

---

### **Node Features**

$$
x_i = \left[\mathbf{e}_i \,\middle|\, \Phi_i^{\text{struct}}\right]
$$

* $x_i$: Final feature vector — `data.x` in PyG `Data` object
* $\mathbf{e}_i$: Embedding (ESM-2 3B 2560-D or ProtBert 1024-D) — loaded from `.pt` files in `data/esm/` or `data/protbert/` via `_load_embedding` in `createdatset.py`
* $\Phi_i^{\text{struct}}$: Structural features — 17-D vector built from CSV columns `SCALAR_COLS + STRUCT_COLS` (RSA, ResFlex, Hydrophobicity, PackingDensity, HSE_up, HSE_down, Poly_bias, Poly_RSA, Poly_Flex, Poly_interaction, BondAngle, sin_phi, cos_phi, sin_psi, cos_psi, sin_omega, cos_omega)
* Concatenation: `torch.cat([emb, struct], dim=-1)` in `__getitem__`. In esm_model: `data.x[:, :2560]` is ESM, `data.x[:, 2560:]` is struct (see [esm_model/train.py:204-205](esm_model/train.py#L204-L205))

---

### **GCN**

$$
\tilde A = \tilde D^{-1/2}(A+I)\tilde D^{-1/2}
$$

* $\tilde A$: Normalized adjacency matrix — applied internally by `GCNConv(..., add_self_loops=True, normalize=True)` at [esm_model/model/gcn.py:8](esm_model/model/gcn.py#L8)
* $\tilde D$: Degree matrix of $A + I$
* $I$: Identity matrix — added by `add_self_loops=True`

$$
H^{(D+1)} = \sigma\Big((1-\alpha)\tilde A H^{(D)} + \alpha H^{(0)}\Big)\Big((1-\beta_D)I + \beta_D W^{(D)}\Big)
$$

* $H^{(D)}$: Node features at layer $D$ — variable `x` in [esm_model/model/gcn.py:38-41](esm_model/model/gcn.py#L38-L41)
* $H^{(0)}$: Initial node features (after input projection) — variable `h0` saved at [esm_model/model/gcn.py:37](esm_model/model/gcn.py#L37)
* $\sigma$: Activation — `F.gelu` at [esm_model/model/gcn.py:14](esm_model/model/gcn.py#L14)
* $W^{(D)}$: Weight matrix — internal to `GCNConv`
* $\alpha$: Initial residual fraction — `self.alpha` (constructor arg `alpha`, tuned per model: 0.10–0.21)
* In code, the per-layer combination is `h = (1 - self.alpha) * h + self.alpha * h0` then `x = x + h` (additional residual) at [esm_model/model/gcn.py:40-41](esm_model/model/gcn.py#L40-L41)

$$
\beta_D = \log\left(\frac{\lambda}{D} + 1\right)
$$

* $\lambda$: Hyperparameter — paper sets $\lambda = 1.5$; in our code the identity-mapping coefficient is folded into `GCNConv`'s weight matrix and not exposed as a separate $\beta_D$ schedule

---

### **BiTCN**

$$
z_t^{(n)} = \text{ReLU}\left(\sum_{j=0}^{k-1} W_j^{(n)} \cdot z_{t + d_b(j-1)}^{(n-1)} + b^{(n)}\right) + z_t^{(n-1)}
$$

* $z_t^{(n)}$: Feature at time $t$, layer $n$ — output of `nn.Conv1d` in [esm_model/model/tcn.py:40,47](esm_model/model/tcn.py#L40)
* $d_b$: Dilation — `dilation=2**i` in [esm_model/model/tcn.py:64](esm_model/model/tcn.py#L64), so block dilations are `[1, 2, 4, 8]`
* $k$: Kernel size — `3` (hard-coded at [esm_model/model/tcn.py:27-28](esm_model/model/tcn.py#L27-L28))
* $W_j^{(n)}, b^{(n)}$: weights and bias of `self.conv1` / `self.conv2`
* Residual `+ z_t^{(n-1)}`: implemented as `return x + y` at [esm_model/model/tcn.py:64](esm_model/model/tcn.py#L64); when `in_ch != out_ch`, `x` is first projected via `self.res = nn.Linear(in_ch, out_ch)`
* Activation in code is `F.gelu` (not ReLU) — same role
* Bidirectional output: forward branch `self.f` + reversed branch `self.b` concatenated, see `BiTCN.forward` in [esm_model/model/tcn.py:91-119](esm_model/model/tcn.py#L91-L119)

---

### **Focal Loss**

$$
p_i = P(y_i = 1 \mid x_i)
$$

* $p_i$: Predicted probability — `p = logp.exp()` at [esm_model/train.py:132](esm_model/train.py#L132), where `logp = torch.log_softmax(logits, dim=1)[:, 1]`

$$
\mathcal{L}_i^{\text{pos}} = -\alpha_{\text{pos}} (1-p_i)^\gamma \log p_i
$$
$$
\mathcal{L}_i^{\text{neg}} = -p_i^\gamma \log(1-p_i)
$$

* $\alpha_{\text{pos}}$: Class weight — `alpha_pos = neg / pos` from `compute_dataset_alpha` in [esm_model/train.py:106-115](esm_model/train.py#L106-L115); ≈ 5.5 for this dataset
* $\gamma$: Focusing parameter — argument `gamma` (defaults to 2.0), tuned via `hp["focal_gamma"]` in $[1.0, 3.0]$
* `loss_pos = -alpha_pos * ((1 - p[pos]) ** gamma) * logp[pos]` at [esm_model/train.py:135](esm_model/train.py#L135)
* `loss_neg = -(p[neg] ** gamma) * torch.log1p(-p[neg])` at [esm_model/train.py:136](esm_model/train.py#L136)
* Reduction: `(loss_pos.sum() + loss_neg.sum()) / targets.numel()` at [esm_model/train.py:137](esm_model/train.py#L137)

---

### **Soft MCC**

$$
TP_s = \sum_i p_i y_i,\quad FP_s = \sum_i p_i (1-y_i)
$$
$$
TN_s = \sum_i (1-p_i)(1-y_i),\quad FN_s = \sum_i (1-p_i)y_i
$$

* $TP_s, FP_s, TN_s, FN_s$: Soft confusion values — variables `tp, fp, tn, fn` in `soft_mcc` at [esm_model/train.py:118-127](esm_model/train.py#L118-L127)
* $p_i$: `p = torch.softmax(logits, dim=1)[:, 1]`
* $y_i$: `y = targets.float()`

$$
\text{MCC}_s = \frac{TP_s\, TN_s - FP_s\, FN_s}{\sqrt{(TP_s+FP_s)(TP_s+FN_s)(TN_s+FP_s)(TN_s+FN_s) + \varepsilon}}
$$

* $\varepsilon$: numerical stabilizer — `eps=1e-7` at [esm_model/train.py:118,126](esm_model/train.py#L118)
* `num = tp * tn - fp * fn`, `den = torch.sqrt(... + eps)`, `return num / den` at [esm_model/train.py:125-127](esm_model/train.py#L125-L127)

---

### **Final Loss**

$$
\mathcal{L} = \mathcal{L}_{\text{focal}} - \lambda_{\text{MCC}} \cdot \text{MCC}_s
$$

* $\lambda_{\text{MCC}}$: Weight for MCC term — argument `lambda_mcc` (default 0.0), tuned via `hp["lambda_mcc"]` in $[0.0, 2.0]$; best models use 1.15-1.95
* `if lambda_mcc > 0.0: loss = loss - lambda_mcc * soft_mcc(logits, targets)` at [esm_model/train.py:138-139](esm_model/train.py#L138-L139)

---

### **AdamW**

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

* $m_t$: First moment (mean) — internal to `torch.optim.AdamW`
* $v_t$: Second moment (variance) — internal to `torch.optim.AdamW`
* $g_t$: Gradient — produced by `scaler.scale(loss).backward()` at [esm_model/train.py:225](esm_model/train.py#L225)
* $\beta_1, \beta_2$: PyTorch defaults `(0.9, 0.999)`
* Optimizer instantiated as `optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])` at [esm_model/train.py:189-193](esm_model/train.py#L189-L193)

---

### **LR Schedule**

$$
\eta_e =
\begin{cases}
\eta_0 \cdot \frac{e+1}{W}, & e < W \\
\eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos(\cdot)\right), & e \ge W
\end{cases}
$$

* $\eta_e$: Learning rate at epoch $e$ — `optimizer.param_groups[0]["lr"]`
* $\eta_0$: Initial LR — `hp["lr"]`
* $\eta_{\min}$: floor — `eta_min=1e-6` at [esm_model/train.py:196](esm_model/train.py#L196)
* $W$: Warmup epochs — `WARMUP_EPOCHS = 5` at [esm_model/train.py:27](esm_model/train.py#L27)
* Implemented as two schedulers: `warmup = LambdaLR(optimizer, lambda e: min(1.0, (e+1)/WARMUP_EPOCHS))` and `cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)` at [esm_model/train.py:195-196](esm_model/train.py#L195-L196)
* Switched per epoch: `if epoch < WARMUP_EPOCHS: warmup.step() else: cosine.step()` at [esm_model/train.py:236-239](esm_model/train.py#L236-L239)

---

### **Gradient Clipping**

$$
g \leftarrow g \cdot \min\left(1,\ \frac{1}{\|g\|_2}\right)
$$

* $g$: Gradient vector — concatenation of all parameter `.grad` tensors
* Implemented as `torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)` at [esm_model/train.py:227](esm_model/train.py#L227); the `1.0` is the max-norm threshold
* Called between `scaler.unscale_(optimizer)` and `scaler.step(optimizer)` so clipping operates on the unscaled gradient

---

### **Metrics**

$$
TP, FP, TN, FN
$$

* $TP$: True Positive — predicted 1, label 1
* $FP$: False Positive — predicted 1, label 0
* $TN$: True Negative — predicted 0, label 0
* $FN$: False Negative — predicted 0, label 1
* Computed implicitly inside `sklearn.metrics.precision_score`, `recall_score`, `f1_score`, `matthews_corrcoef`, `accuracy_score` after thresholding `preds = (probs >= threshold).astype(int)` in `metrics(...)` at [esm_model/test.py:81-92](esm_model/test.py#L81-L92)
* Threshold-independent metrics: `roc_auc_score(labels, probs)` and `average_precision_score(labels, probs)` in the same function
