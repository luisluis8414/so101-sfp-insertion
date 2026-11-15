# Episode editor

Rewatch the episodes of a LeRobot v3.0 dataset with all cameras and the joint curves (measured position and leader target), rate them, trim them and mark them for deletion. Then export a cleaned copy.

```bash
python server.py ~/.cache/huggingface/lerobot/local/my_demos       # opens http://127.0.0.1:8765
python export_edits.py ~/.cache/huggingface/lerobot/local/my_demos \
    ~/.cache/huggingface/lerobot/local/my_demos.edits.json --repo-id local/my_demos_r4 --min-rating 4
```

The editor needs pandas and pyarrow, the export a LeRobot environment.

Every change is saved immediately to `<dataset>.edits.json` next to the dataset. The dataset itself is never modified. The export drops deleted episodes (and, with `--min-rating`, worse ones; `--drop-unrated` also drops unrated ones), applies the trims and re-encodes the videos through LeRobot's dataset writer. Ratings, notes and the source episode of every kept episode are stored in `meta/episode_labels.json` of the new dataset. DAgger datasets have an extra `intervention` feature: export them with `--drop-feature intervention` to merge them with demonstrations.

| Key | |
|---|---|
| Space | play / pause |
| ← → | one frame (Shift: one second) |
| I / O | set start / end of the trim |
| [ ] | jump to start / end |
| 1 – 5, 0 | rate, clear the rating |
| X | mark for deletion |
| N / ↓, P / ↑ | next / previous episode |
| Esc | leave a text field |

Click or drag in the joint plot to seek, and drag the trim edges.
