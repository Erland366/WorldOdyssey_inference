# Local task inputs

Place WorldOdyssey-format task folders here when using `scripts/submit_worldodyssey_task.py` or the checked-in YAML
configs. Each task folder must contain `task.json` and the frame files referenced by that JSON.

The default single-task path is `inputs/move_bookmark`. Parent-directory batch configs use `inputs` and scan its
direct child folders for `task.json`.
