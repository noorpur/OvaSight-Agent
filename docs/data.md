# Data

This project is built for the MMOTU ovarian tumor ultrasound dataset. MMOTU includes 2D ultrasound and contrast-enhanced ultrasound images with pixel-wise semantic annotations and global category labels [1]. The original release describes 1469 2D ultrasound images and 170 CEUS images [1].

## Why the raw data is not included

I do not include raw medical images in this repository. Even when a dataset is public, redistribution terms can vary by source, and it is better research hygiene to make users obtain the data from the official authors or approved mirrors.

## Expected local layout

```text
data/raw/MMOTU/
├── images/
└── masks/
```

The agent can also handle less tidy folder structures by recursively searching for common image and mask naming patterns.

## What gets generated

After preparation, the agent writes:

```text
data/processed/mmotu/images/
data/processed/mmotu/masks/
results/splits.csv
results/data_audit.csv
results/data_audit.json
```

## Data checks

The audit stage checks:

- image and mask pairing
- corrupted files
- image and mask dimensions
- blank masks
- near-empty masks
- duplicate-looking image hashes
- total usable cases

These checks are boring in the same way seatbelts are boring: the whole experiment depends on them.
