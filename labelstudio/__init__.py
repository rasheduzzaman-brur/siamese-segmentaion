"""Label Studio integration for Siamese (reference + inspected) annotation.

- config.py        labelling template: reference and inspected image side by
                   side; box / polygon / brush labels per defect class on the
                   inspected image; a "defect / good" pair choice
- tasks.py         images (+ existing COCO / annotator labels) -> Label Studio tasks
- setup_project.py create + configure a project through the REST API
- pairing.py       pairs references + defect images uploaded with Label Studio's
                   own Import (inbox project -> pairs project, by file name)
- uploader.py      browser upload page: template + raw images -> data/ + tasks
- export_coco.py   Label Studio JSON export -> COCO + reference_image splits
- ml_backend.py    model-assisted pre-labelling backend for Label Studio
- start.sh         run Label Studio with local file serving for data/

See README.md "Annotating with Label Studio".
"""
