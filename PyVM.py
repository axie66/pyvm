'''
python3 -m py_compile <filename.py>     -->     compiles into bytecode

Note: this code in this file only works with Python 3.7 
(bytecode is extremely implementation-dependent)
'''

import dis
import marshal
import sys
import collections
import operator
import types
import inspect

DEBUG=False

file = ''
if not file:
    file = input('Which file should I run? (see programs folder) ')

compiledFile = f'programs/__pycache__/{file}.cpython-37.pyc'

with open(compiledFile, 'rb') as f:
    # Not sure if this always works; specifically, some wonky stuff is
    # going on with bit_field. See here for more: https://www.python.org/dev/peps/pep-0552/
    magicNumber = f.read(4)             # every cpython file has an associated magic number
    bitField = f.read(4)                # does something but i don't know what
    timestamp = f.read(4)               # time when code was compiled
    size = f.read(4)                    # size of bytecode
    code = marshal.loads(f.read())      # the bytecode, as a code object

'''
Code objects contain all the data & metadata needed to run a Python routine.
CPython uses these objects to run your code in the CPython virtual machine 
(written in C, an objectively bad language). Today, we're going to use them to 
run your code in the Python virtual machine!

The following are attributes of code objects:

co_argcount:        num of arguments
co_code:            bytecode in binary form
co_consts:          tuple containing all constant (immutable) values used by function
co_filename:        name of the file being run
co_firstlineno:     first line of Python source code (used in exception tracebacks)
co_flags:           See here: https://docs.python.org/3/library/inspect.html#inspect-module-co-flags
co_cellvars:        tuple containing names of all vars in a function also used in a 
                    nested function
co_freevars:        tuple containing names of all vars in a function defined in an 
                    enclosing function
co_kwonlyargcount:  number of keyword-only args
co_lnotab:          binary encoded mapping of source code line numbers to bytecode indices
co_name:            name of the routine
co_names:           tuple containing names of non-argument vars
                    ===> *_GLOBAL, *_NAME
co_nlocals:         number of local vars
co_stacksize:       stack space required for virtual machine (not really a concern for us)
co_varnames:        tuple containing names of arguments + local vars 
                    ===> *_FAST
'''

print('\n****************************************************************\n')
dis.show_code(code)
print('\n****************************************************************\n')
dis.dis(code)
print('\n****************************************************************\n')

###############################################################################
# Classes
###############################################################################

StackFrame = collections.namedtuple('StackFrame', 
                                    ('code', 'stack', 'pc', 'localVars'))

Block = collections.namedtuple('Block', 
                               ('type', 'start', 'end'))

class PyVMError(Exception):
    pass

###############################################################################
# Helper functions
###############################################################################

def dbg_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def popN(S, n=1):
    '''
    Pops last n elements from S and returns them (in the order they were in S)
    '''
    if len(S) < n:
        raise PyVMError("Not enough items on stack!")
    if n < 0:
        raise ValueError("Can't pop negative items from stack!")
    lastN = S[-n:]
    del S[-n:]
    return lastN

def findBlock(blockStack, blockType):
    '''
    Finds the first blockType block in the block stack
    '''
    startIndex = len(blockStack) - 1
    for i in range(startIndex, -1, -1):
        if blockStack[i].type == blockType:
            return i
    else:
        raise PyVMError(f'No {blockType} block in block stack')

def findBlockWithException(blockStack, exception):
    '''
    Finds the first block that matches exception
    '''
    startIndex = len(blockStack) - 1
    for i in range(startIndex, -1, -1):
        if (blockStack[i].type == 'exception' and 
           isinstance(exception, type(blockStack[i].exception))):
            return i
    else:
        raise exception

UNARY_OPS = {
    'UNARY_POSITIVE': operator.pos,
    'UNARY_NEGATIVE': operator.neg,
    'UNARY_NOT': operator.not_,
    'UNARY_INVERT': operator.invert,
    'GET_ITER': iter
}

BINARY_OPS = {
    'BINARY_ADD': operator.add,
    'BINARY_SUBTRACT': operator.sub,
    'BINARY_MULTIPLY': operator.mul,
    'BINARY_TRUE_DIVIDE': operator.truediv,
    'BINARY_FLOOR_DIVIDE': operator.floordiv,
    'BINARY_MODULO': operator.mod,
    'BINARY_POWER': operator.pow,
    'BINARY_MATRIX_MULTIPLY': operator.matmul,
    'BINARY_SUBSCR': operator.getitem,
    'BINARY_LSHIFT': operator.lshift,
    'BINARY_RSHIFT': operator.rshift,
    'BINARY_AND': operator.and_,
    'BINARY_XOR': operator.xor,
    'BINARY_OR': operator.or_
}

COMPARE_OPS = {
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '>=': operator.ge,
    'in': lambda x, y: x in y,
    'not in': lambda x, y: x not in y,
    'is': operator.is_,
    'is not': operator.is_not,
    'exception match': lambda x, y: issubclass(x, Exception) and issubclass(x, y)
}

###############################################################################
# The heart of the PyVM!!!
###############################################################################

def run(code : code):
    '''
    Takes in a code object and runs its bytecode
    '''
    lastException = None
    bytecode = code.co_code # bytecode, stored as bytes
    pc = 0                  # program counter
    stack = []              # runtime stack
    localVars = dict()      # mapping of local var names to their values
    globalVars = dict()     # mapping of global var names to their values
    callStack = []          # stack containing function frames
    blockStack = []         # stack containing blocks
    builtins = vars(__builtins__)
    
    while(True):
        instruction, arg = bytecode[pc:pc+2]
        opname = dis.opname[instruction]
        
        dbg_print(f'[{pc}]   Instruction:{instruction} ({opname})   Arg:{arg}')

        pc += 2

        if DEBUG:
            x = input('Whaddya wanna do? ').lower()
            while x != 'q':
                if x.startswith('l'):
                    print(localVars)
                elif x.startswith('g'):
                    print(globalVars)
                elif x.startswith('s'):
                    print(stack)
                elif x.startswith('c'):
                    print(callStack)
                elif x.startswith('b'):
                    print(blockStack)
                elif x == 'dump':
                    print('Stack:', stack)
                    print('Locals:', localVars)
                    print('Globals:', globalVars)
                    print('Call Stack:', callStack)
                    print('Block Stack:', blockStack)
                x = input('Whaddya wanna do? ').lower()


        '''
        There are a lot of Python bytecode instructions - we're only gonna 
        implement a subset of them, partly because I'm lazy and partly because
        some can only be implemented in a low-level language like C.

        Here is the full list: https://docs.python.org/3.7/lib/bytecodes.html

        Sidenote: All Python bytecode instructions are actually two bytes long.
        Hence, it would be more appropriate to call it Python "wordcode."
        '''

        try:
###############################################################################
# Simple Stack Manipulations 
###############################################################################

            if opname == 'NOP':
                '''Do literally nothing (used for alignment optimizations)'''
                pass

            elif opname == 'POP_TOP':
                '''Yeet the top of stack'''
                stack.pop()

            elif opname == 'ROT_TWO':
                '''S, x, y  ===>  S, y, x'''
                if (len(stack) < 2):
                    raise PyVMError('Not enough items on stack')
                second, top = popN(stack, 2)
                stack.append(top)
                stack.append(second)
            
            elif opname == 'ROT_THREE':
                '''S, x, y, z  ===>  S, z, x, y'''
                if (len(stack) < 3):
                    raise PyVMError('Not enough items on stack')
                top = stack.pop()
                stack.insert(-2, top)
            
            elif opname == 'ROT_FOUR':
                '''S, x, y, z, w  ===>  S, w, x, y, z'''
                if (len(stack) < 4):
                    raise PyVMError('Not enough items on stack')
                top = stack.pop()
                stack.insert(-3, top)
            
            elif opname == 'DUP_TOP':
                '''S, x  ===>  S, x, x'''
                if (len(stack) < 1):
                    raise PyVMError('Not enough items on stack')
                top = stack[-1]
                stack.append(top)
            
            elif opname == 'DUP_TOP_TWO':
                '''S, x, y  ===>  S, x, y, x, y'''
                if (len(stack) < 2):
                    raise PyVMError('Not enough items on stack')
                top2 = stack[-2:]
                stack += top2

    ###############################################################################
    # Unary Operations
    ###############################################################################

            elif opname in UNARY_OPS:
                '''Perform specified unary operation (see UNARY_OPS dict)'''
                top = stack.pop()
                op = UNARY_OPS.get(opname)
                if op:
                    stack.append(op(top))
                else:
                    raise PyVMError('Bad unary operator')
            
    ###############################################################################
    # Binary Operations
    ###############################################################################

            elif opname.startswith('BINARY'):
                '''Perform specified binary operation (see BINARY_OPS dict)'''
                second, top = popN(stack, 2)
                op = BINARY_OPS.get(opname)
                if op:
                    stack.append(op(second, top))
                else:
                    raise PyVMError('Bad binary operator')

            elif opname == 'COMPARE_OP':
                '''Perform specified comparison (see COMPARE_OPS dict)''' 
                cmp = dis.cmp_op[arg]
                second, top = popN(stack, 2)
                op = COMPARE_OPS.get(cmp)
                if op:
                    stack.append(op(second, top))
                else:
                    raise PyVMError('Bad comparison operator')

    ###############################################################################
    # Inplace Operations
    ###############################################################################

            # We (attempt to) modify the object on the stack itself.

            elif opname.startswith('INPLACE'):
                '''Performs specified inplace operation'''
                top = stack.pop()
                if opname == 'INPLACE_POWER':
                    stack[-1] **= top
                elif opname == 'INPLACE_MULTIPLY':
                    stack[-1] *= top
                elif opname == 'INPLACE_MATRIX_MLUTIPLY':
                    stack[-1] @= top    
                elif opname == 'INPLACE_FLOOR_DIVIDE':
                    stack[-1] //= top
                elif opname == 'INPLACE_TRUE_DVIIDE':
                    stack[-1] /= top
                elif opname == 'INPLACE_MODULO':
                    stack[-1] %= top
                elif opname == 'INPLACE_ADD':
                    stack[-1] += top
                elif opname == 'INPLACE_SUBTRACT':
                    stack[-1] -= top
                elif opname == 'INPLACE_LSHIFT':
                    stack[-1] <<= top
                elif opname == 'INPLACE_RSHIFT':
                    stack[-1] >>= top
                elif opname == 'INPLACE_AND':
                    stack[-1] &= top
                elif opname == 'INPLACE_XOR':
                    stack[-1] ^= top
                elif opname == 'INPLACE_OR':
                    stack[-1] |= top

            elif opname == 'STORE_SUBSCR':
                '''
                Stack effect:  S, x, y, z  ===>  S    
                Side effect:   y[z] = x
                '''
                third, second, top = popN(stack, 3)
                second[top] = third
            
            elif opname == 'DELETE_SUBSCR':
                '''
                Stack effect: S, x, y  ===>  S
                Side effect: del x[y]
                '''
                second, top = popN(stack, 2)
                del second[top]

    ###############################################################################
    # Variables
    ###############################################################################
            
            ###########################################
            # Basic
            # Can access global or local namespaces
            ###########################################

            elif opname == 'STORE_NAME':
                '''
                S, x  ===>  S
                Stores value of x in local variable
                '''
                name = code.co_names[arg]
                localVars[name] = stack.pop()
                if len(stack) == 0:
                    globalVars[name] = localVars[name]
            
            elif opname == 'DELETE_NAME':
                '''
                Deletes local variable
                '''
                name = code.co_names[arg]
                if name in localVars:
                    del localVars[name]
                else:
                    raise PyVMError('Local var not found')

            elif opname == 'LOAD_NAME':
                '''
                S  ===>  S, x
                Pushes local variable onto the stack
                '''
                name = code.co_names[arg]
                if name in localVars:
                    stack.append(localVars[name])
                elif name in globalVars:
                    stack.append(globalVars[name])
                elif name in builtins:
                    stack.append(builtins[name])
                else:
                    raise PyVMError('Local var not found')

            ###########################################
            # Globals
            # Accesses global namespace
            ###########################################

            elif opname == 'STORE_GLOBAL':
                name = code.co_names[arg]
                globalVars[name] = stack.pop()
            
            elif opname == 'DELETE_GLOBAL':
                name = code.co_names[arg]
                if name in globalVars:
                    del globalVars[name]
                elif name in builtins:
                    del builtins[name]
                else:
                    raise PyVMError(f'Unbound global variable: {name}')

            elif opname == 'LOAD_GLOBAL':
                name = code.co_names[arg]
                if name in globalVars:
                    stack.append(globalVars[name])
                elif name in builtins:
                    stack.append(builtins[name])
                else:
                    raise PyVMError(f'Unbound global variable: {name}')

            ###########################################
            # Fast(?)
            # Accesses local namespace only
            ###########################################

            elif opname == 'STORE_FAST':
                name = code.co_varnames[arg]
                localVars[name] = stack.pop()
            
            elif opname == 'DELETE_FAST':
                name = code.co_varnames[arg]
                if name in localVars:
                    del localVars[name]
                else:
                    raise PyVMError(f'Unbound local variable: {name}')

            elif opname == 'LOAD_FAST':
                name = code.co_varnames[arg]
                if name in localVars:
                    stack.append(localVars[name])
                else:
                    raise PyVMError(f'Unbound local variable: {name}')

            ###########################################
            # Attributes
            # Accesses attributes of elem @ top of stack
            ###########################################

            elif opname == 'STORE_ATTR':
                name = code.co_names[arg]
                second, top = popN(stack, 2)
                setattr(top, name, second)
            
            elif opname == 'DELETE_ATTR':
                name = code.co_names[arg]
                top = stack.pop()
                delattr(top, name)

            elif opname == 'LOAD_ATTR':
                name = code.co_names[arg]
                top = stack.pop()
                stack.append(getattr(top, name))

            ###########################################
            # Deref (???)
            # Seems to deal mainly with closures
            ###########################################

            elif opname == 'STORE_DEREF':
                pass

            elif opname == 'DELETE_DEREF':
                pass

            elif opname == 'LOAD_DEREF':
                pass

            elif opname == 'LOAD_CLASSDEREF':
                pass

            ###########################################
            # Other Loads
            ###########################################
            
            elif opname == 'LOAD_CONST':
                stack.append(code.co_consts[arg])

            elif opname == 'LOAD_CLOSURE':
                pass
            
    ###############################################################################
    # Control Flow
    ###############################################################################

            elif opname == 'JUMP_FORWARD':
                pc += arg
            
            elif opname == 'POP_JUMP_IF_TRUE':
                if stack.pop():
                    pc = arg
            
            elif opname == 'POP_JUMP_IF_FALSE':
                if not stack.pop():
                    pc = arg
            
            elif opname == 'JUMP_IF_TRUE_OR_POP':
                if stack[-1]:
                    pc = arg
                else:
                    stack.pop()
            
            elif opname == 'JUMP_IF_FALSE_OR_POP':
                if not stack[-1]:
                    pc = arg
                else:
                    stack.pop()
            
            elif opname == 'JUMP_ABSOLUTE':
                pc = arg
            
            elif opname == 'FOR_ITER':
                '''
                TOS is an iterator. Call its __next__() method. If this yields a 
                new value, push it on the stack (leaving the iterator below it). 
                If the iterator indicates it is exhausted, TOS is popped and the 
                byte code counter is incremented by delta.
                '''
                try:
                    stack.append(stack[-1].__next__())
                except StopIteration:
                    stack.pop()
                    pc += arg

            elif opname == 'SETUP_LOOP':
                blockStack.append(Block('loop', pc, pc + arg))

            elif opname == 'BREAK_LOOP':
                blockIndex = findBlock(blockStack, 'loop')
                blocks = popN(blockStack, len(blockStack) - blockIndex)
                loopBlock = blocks[0]
                pc = loopBlock.end
                
            elif opname == 'CONTINUE_LOOP':
                blockIndex = findBlock(blockStack, 'loop')
                loopBlock = blockStack[blockIndex]
                popN(blockStack, len(blockStack) - blockIndex - 1)
                pc = loopBlock.start

    ###############################################################################
    # Exception
    # Incomplete and very broken
    ###############################################################################

            elif opname == 'POP_BLOCK':
                '''
                Removes one block from the block stack. Per frame, there is a 
                stack of blocks, denoting nested loops, try statements, and such.
                '''
                blockStack.pop()

            elif opname == 'SETUP_EXCEPT':
                '''
                Pushes a try block from a try-except clause onto the block 
                stack. delta points to the first except block.
                '''
                blockStack.append(Block('try', pc, pc + arg))

            elif opname == 'POP_EXCEPT':
                '''
                Removes one block from the block stack. The popped block must 
                be an exception handler block, as implicitly created when 
                entering an except handler. In addition to popping extraneous 
                values from the frame stack, the last three popped values are 
                used to restore the exception state.
                '''
                assert blockStack.pop().type == 'try'

            
            elif opname == 'SETUP_FINALLY':
                '''
                Pushes a try block from a try-except clause onto the block 
                stack. delta points to the finally block.
                '''
                blockStack.append(Block('try', pc, pc + arg))

            elif opname == 'END_FINALLY':
                '''
                Terminates a finally clause. The interpreter recalls whether 
                the exception has to be re-raised, or whether the function 
                returns, and continues with the outer-next block.
                '''
                assert blockStack.pop().type == 'try'

            elif opname == 'RAISE_VARARGS':
                pass

    ###############################################################################
    # Strings
    ###############################################################################

            elif opname == 'BUILD_STRING':
                strs = popN(stack, arg)
                stack.append(''.join(strs))

            elif opname == 'FORMAT_VALUE':
                pass

    ###############################################################################
    # Lists
    ###############################################################################

            elif opname == 'LIST_APPEND': # for list comps
                pass

            elif opname == 'BUILD_LIST':
                stack.append(popN(stack, arg))

    ###############################################################################
    # Tuples
    ###############################################################################

            elif opname == 'BUILD_TUPLE':
                stack.append(tuple(popN(stack, arg)))

    ###############################################################################
    # Sets
    ###############################################################################

            elif opname == 'SET_ADD': # for set comps
                pass

            elif opname == 'BUILD_SET':
                stack.append(set(popN(stack, arg)))

    ###############################################################################
    # Maps (Dicts)
    ###############################################################################

            elif opname == 'MAP_ADD': # for dict comps
                pass

            elif opname == 'BUILD_MAP':
                items = popN(stack, 2 * arg)
                stack.append({items[i]: items[i+1] for i in range(0, arg, 2)})

            elif opname == 'BUILD_CONST_KEY_MAP':
                keys = stack.pop()
                items = popN(stack, arg)
                stack.append(dict(zip(keys, items)))

    ###############################################################################
    # Functions
    ###############################################################################

            elif opname == 'RETURN_VALUE':
                if callStack:
                    retVal = stack[0]
                    code, stack, pc, localVars = callStack.pop()
                    stack.append(retVal)
                    bytecode = code.co_code
                else:
                    assert len(stack) == 1
                    return stack[0]

            elif opname == 'CALL_FUNCTION':
                args = popN(stack, arg)
                fn = stack.pop()
                name = getattr(fn, '__qualname__', None)
                if name and name in builtins:
                    stack.append(builtins[name](*args))
                else: # not a builtin function
                    callStack.append(StackFrame(code, stack, pc, localVars))
                    code = fn
                    bytecode = code.co_code
                    stack = []
                    pc = 0
                    localVars = dict(zip(code.co_varnames, args))
            
            elif opname == 'MAKE_FUNCTION':
                if arg == 0:
                    stack.pop()

            elif opname == 'CALL_FUNCTION_KW':
                pass
            
            elif opname == 'CALL_FUNCTION_EX':
                pass

            elif opname == 'UNPACK_SEQUENCE':
                pass
            
            elif opname == 'UNPACK_EX':
                pass
            
            elif opname == 'LOAD_METHOD':
                name = code.co_names[arg]
                top = stack.pop()
                method = getattr(top, name, None)
                if method:
                    stack.append(method)
                    stack.append(top)
                else:
                    raise PyVMError(f"Couldn't load method: {top}.{name}")
            
            elif opname == 'CALL_METHOD':
                print(stack)
                args = popN(stack, arg)
                method, self = popN(stack, 2)
                stack.append(method(*args))
            
    ###############################################################################
    # Generators
    ###############################################################################

            elif opname == 'YIELD_VALUE':
                pass
            
            elif opname == 'YIELD_FROM':
                pass

    ###############################################################################
    # Classes
    ###############################################################################

            elif opname == 'LOAD_BUILD_CLASS':
                stack.append(__build_class__)

    ###############################################################################
    # Imports
    ###############################################################################

            elif opname == 'IMPORT_STAR':
                module = __import__(stack.pop())
                for key in vars(module):
                    if not key.startswith('_'):
                        obj = module[key]
                        if isinstance(obj, (types.BuiltinFunctionType, 
                                            types.BuiltinMethodType)):
                            localVars[key] = obj
                        elif isinstance(obj, (types.FunctionType, 
                                            types.MethodType)):
                            localVars[key] = obj.__code__
                        else:
                            localVars[key] = obj
                        if len(callStack) == 0:
                            globalVars[key] = localVars[key]

            elif opname == 'IMPORT_NAME':
                moduleName = code.co_names[arg]
                second, top = popN(stack, 2)
                stack.append(__import__(moduleName, fromlist=top, level=second))
            
            elif opname == 'IMPORT_FROM':
                moduleName = code.co_names[arg]
                top = stack.pop()
                stack.append(__import__(moduleName, fromlist=[top]))

    ###############################################################################
    # Misc
    ###############################################################################

            elif opname == 'PRINT_EXPR':
                # used in the repl
                print(stack.pop())
            
            elif opname == 'SETUP_ANNOTATIONS':
                # compiler can just optimize these out
                pass
            
            elif opname == 'BUILD_SLICE':
                args = popN(stack, arg)
                stack.append(slice(*args))
            
            elif opname == 'EXTENDED_ARG':
                pass

            else:
                raise PyVMError(f'Unknown or unimplemented instruction: {opname}')

        except PyVMError as e:
            raise e

        except Exception as e:
            handlerIndex = findBlock(blockStack, 'except')
            handler = blockStack[handlerIndex]
            popN(blockstack, len(blockStack) - handlerIndex - 1)
            pc = block.end
            


run(code)