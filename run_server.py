"""
Launcher for the BioSpark dev server (worktree edition).

Ensures the worktree's backend/ is used — not the main project's stale
__pycache__ or backend package — by manipulating sys.path before uvicorn
imports anything.

Worktree layout:  <main_project>/.claude/worktrees/infallible-blackwell/
So _HERE is 3 levels inside the main project root.
"""
import os
import sys

# Worktree root is always this file's directory
_HERE = os.path.dirname(os.path.abspath(__file__))

# Main project root is 3 levels up: .claude/worktrees/infallible-blackwell → root
_MAIN_PROJECT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))

# Set CWD to the worktree
os.chdir(_HERE)

# Remove the main project root from sys.path so its backend/__pycache__ is never used
_main_proj_norm = os.path.normpath(_MAIN_PROJECT)
sys.path = [
    p for p in sys.path
    if p != "" and os.path.normpath(p) != _main_proj_norm
]

# Put the worktree first so 'backend' resolves here
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Guard required on Windows for multiprocessing spawn (uvicorn --reload)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[
            os.path.join(_HERE, "backend"),
            os.path.join(_HERE, "frontend"),
        ],
    )
