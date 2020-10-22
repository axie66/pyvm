def foo(x):
    i = 0
    while i < x:
        yield i
    yield i + 100

for elem in foo(10):
    print(elem)

lc = [i for i in range(10)]