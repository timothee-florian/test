import numpy as np

def combinationsUniques(items,n):
    "return all unique combinations of n elements"
    if n==0:
        yield[]
    else:
        for i in range(len(items) - n+1):
            for j in combinationsUniques(items[i+1:],n-1):
                yield[items[i]]+j
                
                
def combinationsUnique(items):
    "return all unique combination"
    res = []
    for n in range(1,len(items)+1):
        res+=combinationsUniques(items,n)
    return res                