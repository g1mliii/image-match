"""Compatibility entrypoint for the Flask app.

The implementation lives in ``app_core.py`` so this file stays small while
keeping existing import paths working.
"""

try:
    from . import app_core as _app_impl  # type: ignore
except ImportError:
    import app_core as _app_impl  # type: ignore

# Re-export public names from the implementation module for compatibility.
for _name in dir(_app_impl):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_app_impl, _name)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
