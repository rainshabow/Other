import torch
import torch.nn as nn
from .rbm import RBM

class DBN(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.rbms = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.rbms.append(RBM(layer_sizes[i], layer_sizes[i+1]))

        # classifier (simple MLP)
        modules = []
        for i in range(len(layer_sizes)-1):
            modules.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(layer_sizes[-1], 10))
        self.classifier = nn.Sequential(*modules)

    def forward(self, x):
        return self.classifier(x)

    def pretrain(self, train_loader, epochs=1, lr=1e-3, device='cpu'):
        # unsupervised layer-wise training
        data = None
        for idx, rbm in enumerate(self.rbms):
            rbm.to(device)
            print(f"Pretraining RBM {idx}: {rbm.n_vis}->{rbm.n_hid}")
            for epoch in range(epochs):
                total_loss = 0.0
                n_batches = 0
                for batch in train_loader:
                    # batch can be (x, target)
                    if isinstance(batch, (list, tuple)):
                        x = batch[0]
                    else:
                        x = batch
                    x = x.view(x.size(0), -1).to(device)
                    if idx > 0:
                        # transform data through previous rbms
                        with torch.no_grad():
                            v = x
                            for j in range(idx):
                                prob_h, _ = self.rbms[j].sample_h(v)
                                v = prob_h
                            x_in = v
                    else:
                        x_in = x
                    loss = rbm.contrastive_divergence(x_in, lr=lr)
                    total_loss += loss
                    n_batches += 1
                print(f" RBM {idx} epoch {epoch+1}/{epochs} loss={total_loss/n_batches:.6f}")

    def transfer_rbm_to_classifier(self):
        """Initialize classifier linear layers' weights from pretrained RBM weights.

        This sets the classifier's Linear weight and bias for the prefixed hidden layers
        from corresponding RBM parameters so that supervised fine-tuning starts
        from the pretrained representation.
        """
        # classifier layers are structured as: Linear, ReLU, Linear, ReLU, ... , Linear(out->10)
        # We need to map rbm i (W: n_hid x n_vis) to classifier Linear i weight (out_features=n_hid,in_features=n_vis)
        linear_layers = [m for m in self.classifier.modules() if isinstance(m, nn.Linear)]
        # linear_layers includes all linears including the final output layer; only map first len(rbms) of them
        n_map = min(len(self.rbms), len(linear_layers))
        for i in range(n_map):
            rbm = self.rbms[i]
            lin = linear_layers[i]
            with torch.no_grad():
                if rbm.W.shape == lin.weight.data.shape:
                    lin.weight.data.copy_(rbm.W.data)
                else:
                    # If shapes mismatch, try transposing or reshaping as a fallback
                    try:
                        lin.weight.data.copy_(rbm.W.data)
                    except Exception:
                        print(f"Warning: cannot copy RBM weights to classifier layer {i} due to shape mismatch")
                # copy hidden bias to linear layer bias when sizes match
                if rbm.h_bias.shape[0] == lin.bias.data.shape[0]:
                    lin.bias.data.copy_(rbm.h_bias.data)

    def save(self, path):
        torch.save({'state_dict': self.state_dict(), 'layer_sizes': self.layer_sizes}, path)

    def load(self, path, map_location=None):
        ck = torch.load(path, map_location=map_location)
        if 'layer_sizes' in ck and ck['layer_sizes'] != self.layer_sizes:
            print('Warning: checkpoint layer sizes differ from current DBN instance')
        self.load_state_dict(ck['state_dict'])
