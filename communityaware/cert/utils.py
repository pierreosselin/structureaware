from itertools import product

def triangle_number(n):
    return int((n*(n+1))/2)

def partition(total: int, k: int):
    """Return list of all tuples of length k with non-negative integers that sum to total"""
    return list(filter(lambda x: sum(x)==total, product(*[range(total+1) for _ in range(k)])))