import os

def foo():
    print(1/2)
    try:
        1 / 0
        print("line 7")
    except StopIteration as e:
        print(1)
    except NotImplementedError as e:
        print(2)
    except Exception as e:
        print(3)
        return 42
    finally:
        print('*')
    

print(foo())