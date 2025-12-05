import torch
import torch.nn as nn
from .rbm import RBM

# DBN (Deep Belief Network) 实现说明：
# - 使用多层 RBM 逐层无监督预训练（pretrain），每层训练完成后将其权重用于初始化下一层的输入表示；
# - 预训练完成后，将 RBM 的权重迁移到一个简单的前馈分类器（MLP）的对应线性层，作为监督微调的初始化；
# - 最终通过监督的交叉熵损失（CrossEntropyLoss）对分类器进行 fine-tune。
class DBN(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.rbms = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.rbms.append(RBM(layer_sizes[i], layer_sizes[i+1]))

        # 分类器（简单的多层感知机）
        modules = []
        for i in range(len(layer_sizes)-1):
            modules.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(layer_sizes[-1], 10))
        self.classifier = nn.Sequential(*modules)

    def forward(self, x):
        return self.classifier(x)

    def pretrain(self, train_loader, epochs=1, lr=1e-3, device=torch.device('cpu')):
        # 无监督的逐层预训练（layer-wise training）
        # 每次只训练一个 RBM；当训练第 i 层时，会先用前 i-1 层将原始输入转换成该层的输入表示（hidden probabilities），
        # 然后用该表示去训练第 i 个 RBM（不对前面的 RBM 计算梯度）。
        data = None
        for idx, rbm in enumerate(self.rbms):
            rbm.to(device)
            print(f"预训练 RBM {idx}: {rbm.n_vis}->{rbm.n_hid}")
            for epoch in range(epochs):
                total_loss = 0.0
                n_batches = 0
                for batch in train_loader:
                    # 批数据可能是 (x, target) 的形式
                    if isinstance(batch, (list, tuple)):
                        x = batch[0]
                    else:
                        x = batch
                    x = x.view(x.size(0), -1).to(device)
                    if idx > 0:
                        # 对于非第一层，先通过前面训练好的 RBM 将可见层数据转换为隐藏层概率
                        # 作为当前 RBM 的输入（使用 no_grad 以避免改变前面 RBM 的参数）。
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
                print(f" RBM {idx} 轮次 {epoch+1}/{epochs} 损失={total_loss/n_batches:.6f}")

    def transfer_rbm_to_classifier(self):
        """将预训练得到的 RBM 权重迁移到分类器的线性层中以初始化监督微调。

        细节：classifier 的线性层顺序与 RBM 层一一对应（前几层），因此尝试将第 i 个 RBM 的权重（W）和隐藏偏置（h_bias）
        复制到 classifier 的第 i 个 Linear 的 weight 和 bias 中。若形状不匹配，会发出警告并跳过。
        """
        # classifier 的线性层顺序为：Linear, ReLU, Linear, ReLU, ... , 最后为 Linear(out->10)
        # 需要把第 i 个 RBM 的权重（W: n_hid x n_vis）映射到 classifier 对应的第 i 个 Linear 层
        #（其 weight 的形状为 out_features=n_hid, in_features=n_vis）。
        linear_layers = [m for m in self.classifier.modules() if isinstance(m, nn.Linear)]
        # linear_layers 包含了所有 Linear 层（包括最后的输出层）；这里只映射前面与 RBM 数目相同的层。
        n_map = min(len(self.rbms), len(linear_layers))
        for i in range(n_map):
            rbm = self.rbms[i]
            lin = linear_layers[i]
            with torch.no_grad():
                if rbm.W.shape == lin.weight.data.shape:
                    lin.weight.data.copy_(rbm.W.data)
                else:
                    # 如果形状不匹配，尝试直接复制（若不可行，会捕获异常并发出警告）
                    try:
                        lin.weight.data.copy_(rbm.W.data)
                    except Exception:
                        print(f"警告：无法将第 {i} 个 RBM 权重复制到分类器层（形状不匹配）")
                # 当隐藏偏置长度与 linear 层 bias 长度匹配时，复制偏置
                if rbm.h_bias.shape[0] == lin.bias.data.shape[0]:
                    lin.bias.data.copy_(rbm.h_bias.data)

    def save(self, path):
        torch.save({'state_dict': self.state_dict(), 'layer_sizes': self.layer_sizes}, path)

    def load(self, path, map_location=None):
        ck = torch.load(path, map_location=map_location)
        if 'layer_sizes' in ck and ck['layer_sizes'] != self.layer_sizes:
            print('警告：检查点中的 layer_sizes 与当前 DBN 实例不匹配')
        self.load_state_dict(ck['state_dict'])
