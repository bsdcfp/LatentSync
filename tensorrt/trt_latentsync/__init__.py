
"""
ProductScene package initialization.
Provides common utilities including optional profiler support.
"""

# Optional profiler import - graceful fallback if not available
try:
    from simpleprofiler.profiler import NVTXContext as _NVTXContext
    PROFILER_AVAILABLE = True
    print("[INFO]: SimpleProfiler Available")
    
    # Export the real NVTXContext
    NVTXContext = _NVTXContext
    
except ImportError:
    print("[INFO]: SimpleProfiler not available, profiling disabled")
    PROFILER_AVAILABLE = False
    
    # Create a dummy NVTXContext class for compatibility
    import functools
    
    class _NVTXContextClass:
        """Dummy NVTXContext class that mimics the real SimpleProfiler interface"""
        
        @property
        def enabled(self):
            return False
        
        @property
        def colors_keys(self):
            return []
        
        def range(self, name):
            """Return a dummy context manager for range"""
            return _DummyRange(name)
        
        def __call__(self, name_or_func=None, color=None):
            """Support decorator usage like the original SimpleProfiler"""
            import functools
            if callable(name_or_func):
                # Direct function decoration: @NVTXContext
                @functools.wraps(name_or_func)
                def wrapper(*args, **kwargs):
                    with self.range(name_or_func.__qualname__):
                        return name_or_func(*args, **kwargs)
                return wrapper
            else:
                # Parameter decoration: @NVTXContext("name") or NVTXContext("name")
                if name_or_func is None:
                    # Handle @NVTXContext() case
                    return lambda func: self.__call__(func)
                else:
                    # Handle @NVTXContext("name") case
                    return self.range(name_or_func)

    class _DummyRange:
        """Dummy range context manager"""
        __slots__ = ('_name',)
        
        def __init__(self, name):
            self._name = name
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        
        def __call__(self, func):
            """Support using as decorator: @NVTXContext("name")"""
            import functools
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
    
    # Create the singleton instance
    NVTXContext = _NVTXContextClass()

# Export for easy access
__all__ = ['NVTXContext', 'PROFILER_AVAILABLE']
