import torch
import time
import functools
import contextlib

class Profiler:
    def __init__(self):
        self.events = {}
        self.enabled = False  # HARD DISABLED: Compact profiler is no longer used by request.

    def enable(self):
        """Enable profiling - COMPLETELY DISABLED."""
        pass

    def disable(self):
        """Disable profiling."""
        self.enabled = False

    def start(self, name, stream=None, cpu=False):
        """
        Start recording time for a named section. 
        No-op if profiling is disabled.
        """
        if not self.enabled:
            return

        if name not in self.events:
            self.events[name] = {'start': None, 'elapsed': 0.0, 'count': 0, 'cpu': cpu, 'pending': []}
        
        # We allow multiple pending events for the same name (nested or async)
        # though standard usage is usually consecutive.
        
        if cpu:
            start_event = time.time()
        else:
            start_event = torch.cuda.Event(enable_timing=True)
            if stream is not None:
                start_event.record(stream)
            else:
                start_event.record()
        
        self.events[name]['pending'].append(start_event)

    def stop(self, name, stream=None, cpu=False):
        """
        Stop recording time for a named section.
        No-op if profiling is disabled.
        """
        if not self.enabled:
            return

        if name not in self.events:
            return
        if not self.events[name]['pending']:
            return

        if cpu:
            end_event = time.time()
        else:
            end_event = torch.cuda.Event(enable_timing=True)
            if stream is not None:
                end_event.record(stream)
            else:
                end_event.record()
        
        start_event = self.events[name]['pending'].pop()
        
        if cpu:
            elapsed = (end_event - start_event) * 1000 # to ms
            self.events[name]['elapsed'] += elapsed
            self.events[name]['count'] += 1
        else:
            # For CUDA, we periodically synchronize to prevent event accumulation leaks
            if 'cuda_pairs' not in self.events[name]:
                self.events[name]['cuda_pairs'] = []
            self.events[name]['cuda_pairs'].append((start_event, end_event))
            
            # Auto-sync every 50 events for this name to prevent memory bloat
            if len(self.events[name]['cuda_pairs']) >= 50:
                self._sync_event(name)

    def _sync_event(self, name):
        """Synchronize and accumulate CUDA events for a specific name."""
        if name not in self.events or self.events[name]['cpu']:
            return
            
        pairs = self.events[name].get('cuda_pairs', [])
        if not pairs:
            return
            
        # Optimize: Batch synchronization if possible (though cuda.Event.elapsed_time needs sync anyway)
        torch.cuda.synchronize()
        for start, end in pairs:
            try:
                self.events[name]['elapsed'] += start.elapsed_time(end)
                self.events[name]['count'] += 1
            except Exception:
                pass # Event might be stale or destroyed
        self.events[name]['cuda_pairs'] = []

    def elapsed_time(self, name):
        """
        Get the total accumulated time for a specific named section.
        Syncs and stores the result.
        """
        if name not in self.events:
            return 0.0, 0.0
        
        # Flush pending CUDA events
        if not self.events[name]['cpu']:
            self._sync_event(name)

        total_time = self.events[name]['elapsed']
        total_count = self.events[name]['count']
        
        avg_time = total_time / total_count if total_count > 0 else 0
        return total_time, avg_time

    def get_all_elapsed_times(self):
        """
        Returns a dictionary of the accumulated elapsed times for each recorded event.
        """
        total_times = {}
        avg_times = {}
        # Synchronize once for all CUDA events
        torch.cuda.synchronize()
        for name in list(self.events.keys()):
            total_times[name], avg_times[name] = self.elapsed_time(name)
        return total_times, avg_times

    def sync(self):
        """
        Manually synchronize all recorded events.
        """
        torch.cuda.synchronize()
        self.get_all_elapsed_times() # flush all events

    def reset(self):
        """
        Reset all stored events.
        """
        # Explicitly destroy events to prevent memory leak
        for name in self.events:
            if 'cuda_pairs' in self.events[name]:
                 # Explicitly delete pairs to ensure they aren't leaking
                 self.events[name]['cuda_pairs'] = []
            self.events[name]['pending'] = []
        self.events = {}

    _instance = None

    @staticmethod
    def instance():
        """
        Singleton pattern to get the instance of the profiler.
        """
        if Profiler._instance is None:
            Profiler._instance = Profiler()
        return Profiler._instance
    
    class ProfileContext:
        def __init__(self, profiler, name, stream=None, cpu=False):
            self.profiler = profiler
            self.name = name
            self.stream = stream
            self.cpu = cpu

        def __enter__(self):
            self.profiler.start(self.name, self.stream, cpu=self.cpu)

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.profiler.stop(self.name, self.stream, cpu=self.cpu)
    
    @staticmethod
    def scope(name, stream=None, cpu=False):
        """
        Create a context manager for profiling a block of code.
        """
        return contextlib.nullcontext()

    @staticmethod
    def prof_func(name, cpu=False):
        """
        Decorator to profile a function using the CudaProfiler.
        """
        def decorator(func):
            return func
        return decorator

def prof_summary(profiler: Profiler, rank = None):
    """
    prof result breakdown - SILENCED
    """
    return []


_torch_profiler = None


def torch_profiler_step():
    global _torch_profiler
    if _torch_profiler is not None:
        _torch_profiler.step()


def set_torch_profiler(profiler):
    global _torch_profiler
    _torch_profiler = profiler