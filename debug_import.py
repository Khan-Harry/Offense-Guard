import sys
import os
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
print(f"CWD: {os.getcwd()}")
print("Path:")
for p in sys.path:
    print(f"  {p}")

try:
    import flask_sqlalchemy
    print("SUCCESS: flask_sqlalchemy imported from:", flask_sqlalchemy.__file__)
except ImportError as e:
    print("FAILURE: Could not import flask_sqlalchemy")
    print(f"Error: {e}")
