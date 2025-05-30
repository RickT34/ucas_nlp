"""A module for passing arguments and receiving results through a pipe. By TR."""

from typing import Any, Callable
import itertools

import tqdm


class ArgPipe:
    """Pass arguments and receive results through a pipe."""

    def __init__(self, /, **kwds):
        self.data = dict(**kwds)

    def __getitem__(self, name: str) -> Any:
        return self.data[name]

    def insert(self, name: str, value: Any) -> None:
        if name not in self.data:
            self.data[name] = value
        else:
            if not isinstance(self.data[name], list) or not isinstance(value, list):
                self.data[name] = value
            else:
                self.data[name].extend(value)

    def set(self, name: str, value: Any):
        self.data[name] = value

    def __str__(self):
        return str(self.data)


class PipeFunctionBase:
    def __init__(self, name):
        self.name = name

    def __call__(self, p: ArgPipe) -> ArgPipe:
        raise NotImplementedError()

    def __or__(self, value: Callable) -> "PipeLine":
        res = PipeLine(self)
        return res | value
    

class PipeFunction(PipeFunctionBase):
    def __init__(self, func, name):
        super().__init__(name)
        self.func = func

    def __call__(self, p: ArgPipe) -> ArgPipe:
        res = self.func(p)
        return res
    

def pipewarp(func):
    """Wrap a function to make it pipe-able."""
    args = func.__code__.co_varnames

    def wrapper(p: ArgPipe):
        res = func(**{k: p[k] for k in args if k in p.data})
        if res is not None:
            if isinstance(res, dict):
                p.data.update(res)
            elif isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], str):
                p.insert(res[0], res[1])
            else:
                p.insert("re_" + func.__name__, res)
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

    def exec(self, /, **kwds: Any) -> dict:
        p = ArgPipe(**kwds)
        p = self(p)
        return p.data

    def __call__(self, p: ArgPipe) -> ArgPipe:
        for func in self.funcs:
            p = func(p)
        return p

    def __str__(self):
        return "PipeLine(" + " | ".join(f.name for f in self.funcs) + ")"
    


class Batch(PipeFunctionBase):
    def __init__(self, forkname: dict[str, str], func: PipeFunctionBase):
        super().__init__(func.name + "_forked")
        self.forkname = forkname
        self.func = func

    def __call__(self, p: ArgPipe) -> ArgPipe:
        keys = tuple(self.forkname.keys())
        for values in tqdm.tqdm(tuple(itertools.product(*[p[k] for k in keys]))):
            for k, v in zip(keys, values):
                p.set(self.forkname[k], v)
            p = self.func(p)
        for k in self.forkname.values():
            del p.data[k]
        return p
    

def debug(func: PipeFunctionBase):
    def debug_wrapper(p: ArgPipe):
        print(f"Executing {func.name} with {p}")
        res = func(p)
        print(f"Result of {func.name}: {res}")
        return res

    return PipeFunction(debug_wrapper, func.name + "_debug")


if __name__ == "__main__":

    @pipewarp
    def add(a, b):
        return a + b

    @pipewarp
    def sub(a, b):
        return a - b

    @pipewarp
    def mul(a, b):
        return a * b

    @pipewarp
    def div(a, b):
        return a / b

    @pipewarp
    def genlist(n):
        return "inputs", list(range(n))

    @pipewarp
    def worker(i):
        return i * 2

    p = genlist | Batch({"inputs": "i"}, worker)
    print(
        p.exec(n=5)
    )  # {'re_add': 5,'re_sub': -1,'re_mul': 6,'re_div': 0.6666666666666666}
