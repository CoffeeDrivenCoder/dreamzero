from pathlib import Path

from groot.vla.data.schema import EmbodimentTag

# Dataset registry: maps embodiment tags to local dataset paths.
# Update these paths to point to your local dataset directories.
# See docs/2025-11-01_DreamZero新机器人数据转换与训练指南.md for dataset conversion instructions.
EMBODIMENT_TAGS_TO_DATASET_PATHS: dict[EmbodimentTag, list[Path]] = {}

DATASET_PATHS_TO_EMBODIMENT_TAGS = {
    path: dataset_tag
    for dataset_tag, dataset_paths in EMBODIMENT_TAGS_TO_DATASET_PATHS.items()
    for path in dataset_paths
}
