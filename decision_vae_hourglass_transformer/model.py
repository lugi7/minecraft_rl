from torch import nn

# Model takes an input of length seq_len (consisting of 3 * seq_len tokens)
# Optionally takes memory as additional input (TransformerXL) to make context length bigger
# Environment runs with 20 fps, so 512 frames equals to 25,6 seconds
# For more complex environments a DNC might be required (which would run every N steps)
# Model outputs predicted actions


class VDT(nn.Module):
    def __init__(self):
        super(VDT, self).__init__()
            
    def forward(self, x, memory=None):
        raise NotImplementedError
    