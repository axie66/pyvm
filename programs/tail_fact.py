def tail_fact(n, acc=0):
    if n == 0:
        return acc
    return tail_fact(n-1, acc*n)