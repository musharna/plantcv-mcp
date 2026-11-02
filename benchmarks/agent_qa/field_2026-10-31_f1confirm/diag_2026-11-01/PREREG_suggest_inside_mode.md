# Pre-registration: `threshold_inside_mode` in `suggest_segmentation`, 2026-11-01

Written and pushed before the check below is run. The warning itself, its
signal and its 0.4 cut-off are unchanged from `PREREG_threshold_inside_mode.md`;
what is new is that `suggest_segmentation` reports it in `polarity.warnings`.

## Claim

`suggest_segmentation` says `threshold_inside_mode` on exactly the cells where
`segment` said it.

## Check

`suggest_check.py` crops each of the 739 held-out cells in `heldout.json` as
`heldout.py` did and calls the real `suggest_segmentation` tool (`a`, `otsu`).
Before a cell counts, its channel's density at the Otsu value must equal the
`density` recorded in `heldout.json` (the crop is the same crop).

## Pass criterion

- **S1:** for all 739 cells, `threshold_inside_mode` in `polarity.warnings`
  equals that cell's `fires` in `heldout.json` (351 OVER, 6 RIGHT, 0 UNDER
  fire there). Any mismatch fails; a missing or duplicate cell fails.

Reported without a gate: how many OVER cells the report recommended `dark`,
not ambiguous, with no other warning (silent before this change).
