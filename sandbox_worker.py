"""
Standalone sandboxed code-execution worker for evaluate_code_execution.py.

Run as a completely separate OS process via subprocess.run() -- NOT via
Python's multiprocessing module. That sidesteps two real problems
multiprocessing has here: (1) the parent process has an initialized CUDA
context from loading the model, and fork()-ing a process that owns a CUDA
context is unsafe; (2) multiprocessing's "spawn" start method re-executes
a script's top-level code in the child, which would reload the entire
model in every one of the ~80 sandboxed runs this eval does, since the
parent script's model-loading code isn't guarded by
`if __name__ == "__main__":`. A plain subprocess avoids both.

Sandboxing approach: rather than deleting dangerous names (open, eval,
exec, compile, ...) off the process-wide `builtins` module, we build a
*restricted copy* of builtins and pass it as the executed code's
`__builtins__` in its exec() globals. This scopes the restriction to only
the untrusted code. Mutating the real builtins module instead breaks
Python's own import machinery -- CPython's import system uses exec/compile
internally to load modules, so deleting those globally means even a
perfectly innocent `import matplotlib` a few lines later raises
NameError: name 'exec' is not defined. (This was caught by testing the
first version of this script locally, not guessed at.)

Protocol: reads the code to execute from stdin, runs it in the restricted
environment, and prints exactly one line to stdout prefixed with
RESULT_MARKER, containing a JSON object:
  {"status": "success" | "syntax_error" | "runtime_error", "error": str|null}
Any other stdout (e.g. print() calls inside the executed code) may appear
on other lines -- the caller looks specifically for the marker line.
"""

import builtins as _builtins_module
import json
import sys

RESULT_MARKER = "###SANDBOX_RESULT###"

BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket", "requests", "urllib",
    "ctypes", "pathlib", "multiprocessing", "threading", "signal",
}

DANGEROUS_BUILTINS = ("open", "eval", "exec", "compile", "input", "exit", "quit")


def main():
    code = sys.stdin.read()

    # Set up matplotlib headless mode BEFORE building the restricted
    # sandbox, using the real, unrestricted import system -- this is our
    # own trusted setup code, not the untrusted generated code.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.show = lambda *a, **k: None
    except ImportError:
        pass

    real_import = _builtins_module.__import__

    def guarded_import(name, *args, **kwargs):
        top_level = name.split(".")[0]
        if top_level in BLOCKED_MODULES:
            raise ImportError(f"import of '{name}' is blocked in this sandbox")
        return real_import(name, *args, **kwargs)

    # A restricted builtins namespace for the EXECUTED CODE ONLY. Copying
    # the full builtins dict keeps normal things (len, range, print, str,
    # etc.) working, while removing/overriding specific dangerous names.
    # The real process-wide `builtins` module is never modified.
    restricted_builtins = dict(vars(_builtins_module))
    for name in DANGEROUS_BUILTINS:
        restricted_builtins.pop(name, None)
    restricted_builtins["__import__"] = guarded_import

    sandbox_globals = {"__builtins__": restricted_builtins, "__name__": "__main__"}

    try:
        compiled = _builtins_module.compile(code, "<generated>", "exec")
        _builtins_module.exec(compiled, sandbox_globals)
        result = {"status": "success", "error": None}
    except SyntaxError as e:
        result = {"status": "syntax_error", "error": f"{type(e).__name__}: {e}"}
    except Exception as e:
        result = {"status": "runtime_error", "error": f"{type(e).__name__}: {e}"}

    print(RESULT_MARKER + json.dumps(result))


if __name__ == "__main__":
    main()
