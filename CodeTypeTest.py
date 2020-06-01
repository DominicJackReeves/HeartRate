from types import CodeType

def sub(a,b):
    return a-b

print(sub(1,2))

payload = b'|\x00|\x01\x17\x00S\x00'





def fix_function(func, payload):
    fn_code = func.__code__
    func.__code__ = CodeType(fn_code.co_argcount,
                             fn_code.co_kwonlyargcount,
                             fn_code.co_nlocals,
                             fn_code.co_stacksize,
                             fn_code.co_flags,
                             payload,
                             fn_code.co_consts,
                             fn_code.co_names,
                             fn_code.co_varnames,
                             fn_code.co_filename,
                             fn_code.co_name,
                             fn_code.co_firstlineno,
                             fn_code.co_lnotab,
                             fn_code.co_freevars,
                             fn_code.co_cellvars,
                             )


fix_function(sub,payload)
sub(1,2)
