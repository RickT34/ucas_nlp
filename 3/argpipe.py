"""A module for passing arguments and receiving results through a pipe. By TR."""

from typing import Any, Callable
import itertools

import tqdm


class ArgPipe:
    """Pass arguments and receive results through a pipe."""

    def __init__(self, /, **kwds):
        self._data = dict(**kwds)
        self.stack = []
        self.temp_keys = set()

    def __getitem__(self, name: str) -> Any:
        return self._data[name]

    def insert(self, name: str, value: Any) -> None:
        if name not in self._data:
            self.temp_keys.add(name)
            self._data[name] = value
        else:
            assert name in self.temp_keys, f"Key {name} is read-only."
            if not isinstance(self._data[name], list) or not isinstance(value, list):
                self._data[name] = value
            else:
                self._data[name].extend(value)

    def push(self):
        self.stack.append(self.temp_keys)
        self.temp_keys = set()
    
    def pop(self):
        d = dict()
        for k in self.temp_keys:
            d[k] = self._data[k]
            del self._data[k]
        self.temp_keys = self.stack.pop()
        return d
    
    def __str__(self):
        return str(self._data)

    @property
    def keys(self):
        return self._data.keys()
    
    # @property
    # def data(self):
    #     return self._data

class PipeFunctionBase:
    def __init__(self, name):
        self.name = name

    def __call__(self, p: ArgPipe) -> ArgPipe:
        raise NotImplementedError()

    def __or__(self, value: Callable) -> "PipeLine":
        res = PipeLine(self)
        return res | value

    def exec(self, /, **kwds: Any):
        p = ArgPipe(**kwds)
        p = self(p)
        return p
    

class PipeFunction(PipeFunctionBase):
    def __init__(self, func, name):
        super().__init__(name)
        self.func = func

    def __call__(self, p: ArgPipe) -> ArgPipe:
        res = self.func(p)
        return res
    
def _process_result(res, default_name:str):
    if res is None:
        return {}
    elif isinstance(res, dict):
        return res
    elif isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], str):
        return {res[0]: res[1]}
    else:
        return {default_name: res}

def pipewarp(func):
    """Wrap a function to make it pipe-able."""
    args = func.__code__.co_varnames

    def wrapper(p: ArgPipe):
        res = func(**{k: p[k] for k in args if k in p.keys})
        d = _process_result(res, "re_" + func.__name__)
        for k, v in d.items():
            p.insert(k, v)
        return p

    return PipeFunction(wrapper, func.__name__)


class PipeLine(PipeFunctionBase):
    """A pipeline of functions that can be executed in sequence."""

    def __init__(self, *funcs: PipeFunctionBase):
        self.funcs: list[PipeFunctionBase] = list(funcs)
        super().__init__(str(self))

    def __or__(self, other: Callable):
        if not isinstance(other, PipeFunctionBase):
            other = pipewarp(other)
        self.funcs.append(other)
        return self


    def __call__(self, p: ArgPipe) -> ArgPipe:
        for func in self.funcs:
            p = func(p)
        return p

    def __str__(self):
        return "PipeLine(" + " | ".join(f.name for f in self.funcs) + "...)"
    


class Batch(PipeFunctionBase):
    def __init__(self, fork: dict[str, str], func: PipeFunctionBase, gather: dict[str, str]):
        super().__init__(func.name + "_batch")
        self.forknames = fork
        self.func = func
        self.gathernames = gather

    def __call__(self, p: ArgPipe) -> ArgPipe:
        keys = tuple(self.forknames.keys())
        for values in tqdm.tqdm(tuple(zip(*[p[k] for k in keys])), desc=self.name):
            p.push()
            for k, v in zip(keys, values):
                p.insert(self.forknames[k], v)
            p = self.func(p)
            red = p.pop()
            for n, k in self.gathernames.items():
                if k in red:
                    p.insert(n, [red[k]])
        return p


def debug(func: PipeFunctionBase):
    def debug_wrapper(p: ArgPipe):
        print(f"Executing {func.name} with {p}")
        res = func(p)
        print(f"Result of {func.name}: {res}")
        return res

    return PipeFunction(debug_wrapper, func.name + "_debug")


