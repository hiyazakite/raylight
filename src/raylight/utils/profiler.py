import cProfile
import pstats
import io
import contextlib
from functools import wraps

@contextlib.contextmanager
def CProfileContext(enabled=True, sort_by='tottime', top_k=5, name="Profile"):
    """
    Context manager for cProfiling a block of code and printing a clean summary
    reporting the top K slowest functions.
    """
    if not enabled:
        yield
        return
        
    pr = cProfile.Profile()
    pr.enable()
    try:
        yield
    finally:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
        ps.print_stats(top_k)
        
        print(f"\n{'='*20} {name} Profiling (Top {top_k} by {sort_by}) {'='*20}")
        # Strip the first few lines of overhead from pstats output for clarity
        lines = s.getvalue().strip().split('\n')
        for line in lines:
            # Filter out some built-ins or completely trivial things if we wanted, 
            # but straight printing is usually best
            print(line)
        print(f"{'='*(42 + len(name))}\n")

def profile_func(enabled=True, sort_by='tottime', top_k=5):
    """
    Decorator for cProfiling a function and printing a clean summary.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)
                
            with CProfileContext(enabled=True, sort_by=sort_by, top_k=top_k, name=func.__qualname__):
                return func(*args, **kwargs)
        return wrapper
    return decorator
