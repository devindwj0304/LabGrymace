"""Test setup shared by the whole build.

The package __init__ imports torch for GPU setup. On a machine whose torch build
does not match its CUDA runtime, that import fails and blocks every test, even
though the categorizer never uses torch. Use the real torch when it imports, and
a small stub when it does not, so the tests can run either way.
"""
import sys
import types

try:
    import torch  # noqa: F401
except Exception:
    torch = types.ModuleType('torch')
    cuda = types.ModuleType('torch.cuda')
    cuda.is_available = lambda: False
    torch.cuda = cuda
    sys.modules['torch'] = torch
    sys.modules['torch.cuda'] = cuda
