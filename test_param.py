
import torch

class GGMLTensor(torch.Tensor):
    def __init__(self, *args, tensor_type=None, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type

    def __new__(cls, *args, tensor_type=None, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        return new

def test_param_attributes():
    t = torch.randn(10)
    g = GGMLTensor(t, tensor_type="Q8_0")
    g.tensor_type = "Q8_0" # Ensure it's set
    
    print(f"Original Tensor Type: {getattr(g, 'tensor_type', 'MISSING')}")
    
    p = torch.nn.Parameter(g)
    print(f"Parameter Wrapper Type: {getattr(p, 'tensor_type', 'MISSING')}")
    
    p2 = p.to(torch.device("cpu"))
    print(f"Parameter .to() Type: {getattr(p2, 'tensor_type', 'MISSING')}")

    # Test if we can manually set it
    p.tensor_type = "Q8_0"
    print(f"Parameter Manually Set: {getattr(p, 'tensor_type', 'MISSING')}")
    
    p3 = p.to(torch.device("cpu"))
    print(f"Parameter with Attr .to() Type: {getattr(p3, 'tensor_type', 'MISSING')}")

if __name__ == "__main__":
    test_param_attributes()
