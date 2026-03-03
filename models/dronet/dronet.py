import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.nn.modules.utils import _pair    # utility to turn int → (int, int)
# from torchinfo import summary

# SMALL = True

# dronet implementation in pytorch.
class DronetTorch(nn.Module):
    def __init__(self, img_dims, img_channels, output_dim):
        """
        Define model architecture.
        
        ## Arguments

        `img_dim`: image dimensions.

        `img_channels`: Target image channels.

        `output_dim`: Dimension of model output.

        """
        super(DronetTorch, self).__init__()

        # get the device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.img_dims = img_dims
        self.channels = img_channels
        self.output_dim = output_dim
        self.conv_modules = nn.ModuleList()
        self.beta = torch.Tensor([0]).float().to(self.device)

        # Initialize number of samples for hard-mining

        if SMALL:
            self.conv_modules.append(nn.Conv2d(self.channels, 32, (3,3), stride=(2,2), padding=(1,1)))
        else:
            self.conv_modules.append(nn.Conv2d(self.channels, 32, (5,5), stride=(2,2), padding=(2,2)))
        filter_amt = np.array([32,64,128])
        for f in filter_amt:
            x1 = int(f/2) if f!=32 else f
            x2 = f
            self.conv_modules.append(nn.Conv2d(x1, x2, (3,3), stride=(2,2), padding=(1,1)))
            self.conv_modules.append(nn.Conv2d(x2, x2, (3,3), padding=(1,1)))
            self.conv_modules.append(nn.Conv2d(x1, x2, (1,1), stride=(2,2)))
        # create convolutional modules
        self.maxpool1 = nn.MaxPool2d((3,3), (2,2))

        bn_amt = np.array([32,32,32,64,64,128])
        self.bn_modules = nn.ModuleList()
        for i in range(6):
            self.bn_modules.append(nn.BatchNorm2d(bn_amt[i]))

        self.relu_modules = nn.ModuleList()
        for i in range(7):
            self.relu_modules.append(nn.ReLU())
        self.dropout1 = nn.Dropout()

        if SMALL:
            self.linear1 = nn.Linear(2048, output_dim)
            self.linear2 = nn.Linear(2048, output_dim)
        else:
            self.linear1 = nn.Linear(6272, output_dim)
            self.linear2 = nn.Linear(6272, output_dim)
        self.sigmoid1 = nn.Sigmoid()
        self.init_weights()
        self.decay = 0.1

        

    def init_weights(self):
        '''
        intializes weights according to He initialization.

        ## parameters

        None
        '''
        torch.nn.init.kaiming_normal_(self.conv_modules[1].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[2].weight)

        torch.nn.init.kaiming_normal_(self.conv_modules[4].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[5].weight)

        torch.nn.init.kaiming_normal_(self.conv_modules[7].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[8].weight)

    def forward(self, x):
        '''
        forward pass of dronet
        
        ## parameters

        `x`: `Tensor`: The provided input tensor`
        '''
        bn_idx = 0
        conv_idx = 1
        relu_idx = 0

        x = self.conv_modules[0](x)
        x1 = self.maxpool1(x)
        
        for i in range(3):
            x2 = self.bn_modules[bn_idx](x1)
            x2 = self.relu_modules[relu_idx](x2)
            x2 = self.conv_modules[conv_idx](x2)
            x2 = self.bn_modules[bn_idx+1](x2)
            x2 = self.relu_modules[relu_idx+1](x2)
            x2 = self.conv_modules[conv_idx+1](x2)
            x1 = self.conv_modules[conv_idx+2](x1)
            x3 = torch.add(x1,x2)
            x1 = x3
            bn_idx+=2
            relu_idx+=2
            conv_idx+=3

        if SMALL:
            x4 = torch.flatten(x3).reshape(-1, 2048)
        else:
            x4 = torch.flatten(x3).reshape(-1, 6272)
        x4 = self.relu_modules[-1](x4)
        x5 = self.dropout1(x4)

        steer = self.linear1(x5)

        collision = self.linear2(x5)
        collision = self.sigmoid1(collision)

        return steer, collision

    def loss(self, k, steer_true, steer_pred, coll_true, coll_pred):
        '''
        loss function for dronet. Is a weighted sum of hard mined mean square
        error and hard mined binary cross entropy.

        ## parameters

        `k`: `int`: the value for hard mining; the `k` highest losses will be learned first,
        and the others ignored.

        `steer_true`: `Tensor`: the torch tensor for the true steering angles. Is of shape
        `(N,1)`, where `N` is the amount of samples in the batch.

        `steer_pred`: `Tensor`: the torch tensor for the predicted steering angles. Also is of shape
        `(N,1)`.

        `coll_true`: `Tensor`: the torch tensor for the true probabilities of collision. Is of 
        shape `(N,1)`

        `coll_pred`: `Tensor`: the torch tensor for the predicted probabilities of collision.
        Is of shape `(N,1)`
        '''
        # for steering angle
        mse_loss = self.hard_mining_mse(k, steer_true, steer_pred)
        # for collision probability
        bce_loss = self.beta * (self.hard_mining_entropy(k, coll_true, coll_pred))
        return mse_loss + bce_loss

    def hard_mining_mse(self, k, y_true, y_pred):
        '''
        Compute Mean Square Error for steering 
        evaluation and hard-mining for the current batch.

        ### parameters
        
        `k`: `int`: number of samples for hard-mining

        `y_true`: `Tensor`: torch Tensor of the expected steering angles.
        
        `y_pred`: `Tensor`: torch Tensor of the predicted steering angles.
        '''
        loss_steer = (y_true - y_pred)**2

        # hard mining        
        # get value of k that is minimum of batch size or the selected value of k
        k_min = min(k, y_true.shape[0])
        _, indices = torch.topk(loss_steer, k=k_min, dim=0)
        max_loss_steer = torch.gather(loss_steer, dim=0, index=indices)
        # mean square error
        hard_loss_steer = torch.div(torch.sum(max_loss_steer), k_min)
        return hard_loss_steer
        

    def hard_mining_entropy(self, k, y_true, y_pred):
        '''
        computes binary cross entropy for probability collisions and hard-mining.

        ## parameters

        `k`: `int`: number of samples for hard-mining

        `y_true`: `Tensor`: torch Tensor of the expected probabilities of collision.

        `y_pred`: `Tensor`: torch Tensor of the predicted probabilities of collision.
        '''

        loss_coll = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        k_min = min(k, y_true.shape[0])
        _, indices = torch.topk(loss_coll, k=k_min, dim=0)
        max_loss_coll = torch.gather(loss_coll, dim=0, index=indices)
        hard_loss_coll = torch.div(torch.sum(max_loss_coll), k_min)
        return hard_loss_coll
    
def to_py(x):
    """Convert np.generic → Python scalar, recurse into tuples."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, tuple):
        return tuple(to_py(y) for y in x)
    return x


def to_py(x):
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, tuple):     return tuple(to_py(y) for y in x)
    return x

def extract_layer_info(model: nn.Module, input_shape):
    import torch.nn as nn
    from torch.nn.modules.utils import _pair

    x = torch.randn(input_shape)
    info = []
    out_map = {}   # tensor-id -> layer_name
    add_counter = 0

    # helper: turn (N,C,H,W) → (N,H,W,C), leave other lengths untouched
    def reorder4(shape):
        if len(shape) == 4:
            n, c, h, w = shape
            return (n, h, w, c)
        return shape

    # --- monkey‐patch torch.add to catch residual adds ---
    orig_add = torch.add
    def add_hook(a, b):
        nonlocal add_counter
        out = orig_add(a, b)
        name = f"add_{add_counter}"
        add_counter += 1

        rec = {
            'name':       name,
            'type':       'Add',
            'in_shape_0': reorder4(tuple(a.shape)),
            'in_shape_1': reorder4(tuple(b.shape)),
            'out_shape':  reorder4(tuple(out.shape)),
            'input_from': out_map.get(id(a), 'input') + ',' + out_map.get(id(b), 'input'),
        }
        info.append(rec)
        out_map[id(out)] = name
        return out

    torch.add = add_hook
    orig_tensor_add = torch.Tensor.__add__
    def tensor_add_hook(self, other):
        return add_hook(self, other)
    torch.Tensor.__add__ = tensor_add_hook

    # --- module hooks ---
    def make_hook(layer_name):
        def hook(module, inp, out):
            rec = {
                'name':       layer_name,
                'type':       module.__class__.__name__,
                'in_shape':   reorder4(tuple(inp[0].shape)),
                'out_shape':  reorder4(tuple(out.shape)),
                'input_from': out_map.get(id(inp[0]), 'input'),
            }

            if isinstance(module, nn.Conv2d):
                ks = _pair(module.kernel_size)
                st = _pair(module.stride)
                pd = _pair(module.padding)
                dl = _pair(module.dilation)
                rec.update({
                    'in_channels':  module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size':  ks,
                    'stride':       st,
                    'padding':      pd,
                    'dilation':     dl,
                })
            elif isinstance(module, nn.MaxPool2d):
                ks = _pair(module.kernel_size)
                st = _pair(module.stride)
                pd = _pair(module.padding)
                rec.update({
                    'kernel_size': ks,
                    'stride':      st,
                    'padding':     pd,
                })
            elif isinstance(module, nn.Linear):
                rec.update({
                    'in_features':  module.in_features,
                    'out_features': module.out_features,
                })
            elif isinstance(module, nn.ReLU):
                rec['activation'] = 'ReLU'
            elif isinstance(module, nn.Sigmoid):
                rec['activation'] = 'Sigmoid'

            # sanitize
            for k, v in rec.items():
                rec[k] = to_py(v)

            info.append(rec)
            out_map[id(out)] = layer_name
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.Linear, nn.ReLU, nn.Sigmoid)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # --- run the forward pass under our patched add() ---
    model.eval()
    with torch.no_grad():
        model(x)

    # --- restore everything ---
    for h in hooks:
        h.remove()
    torch.add = orig_add
    torch.Tensor.__add__ = orig_tensor_add

    return info


if SMALL:
    model = DronetTorch(img_dims=(112,112), img_channels=3, output_dim=1)
    layer_info = extract_layer_info(model, input_shape=(1,3,112,112))
else:
    model = DronetTorch(img_dims=(224,224), img_channels=3, output_dim=1)
    layer_info = extract_layer_info(model, input_shape=(1,3,224,224))


# Now `layer_info` is a list of dicts; you can print or dump to JSON/CSV:
import json
print(json.dumps(layer_info, indent=2))

# Export to ONNX
dummy_input = torch.randn(1, 3, 112, 112) if SMALL else torch.randn(1, 3, 224, 224)
onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dronet.onnx")
torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['steer', 'collision'])
print(f"Exported ONNX model to {onnx_path}")