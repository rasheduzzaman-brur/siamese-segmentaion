"""Local web annotation tool for the Siamese defect dataset.

    python -m annotator --images data/prepared/images --references data/prepared/reference

Annotations are kept in the same COCO + `reference_image` format that
datasets/dataset.py loads (see annotator/store.py), with boxes for detection
and polygon/brush masks (stored as COCO polygon `segmentation`) for
segmentation, each carrying a class from configs/config.yaml
data.defect_classes.
"""
