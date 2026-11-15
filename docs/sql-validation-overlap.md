# SQL validation overlap

The three SQL datasets share examples. Splitting each source independently by
row number allowed copies of training tasks into validation.

Training startup now reads complete SQL training documents and excludes SQL
validation documents with matching normalized question/answer pairs across any
source. Repeated validation pairs are also excluded. Sampling uses retained row
indices, preserving the original EOS boundaries. Source sampling weights use
only retained token counts. The binary files are not rewritten.

Normalization ignores case and whitespace and applies Unicode NFKC. Schema
is deliberately omitted: this is conservative and can exclude distinct tasks.
This does not detect paraphrases or equivalent queries written differently.
Malformed validation rows are excluded. Training rows with missing questions
blacklist their SQL answers; training rows without identifiable answers stop
startup rather than silently bypassing the check.

The scan adds CPU startup work to each training run and logs retained/excluded
counts. Existing W&B results are unchanged and should be labeled as using the
original overlapping split. Do not compare their SQL loss directly with the
new filtered metric.

For future training from scratch, combine sources, group equivalent tasks,
then assign whole groups to train or validation before writing token files.
A new split cannot make an existing checkpoint forget examples it already saw.
New datasets are useful for additional coverage only after this split issue is
fixed; they can themselves overlap the current sources.

Verified on the existing `sqlgen-data` Modal Volume: 3,716 NSText2SQL tasks,
644 sql-create-context tasks, and 28 additional Text-to-sql-v1 tasks remain
(4,388 total). Source order decides which copy of a repeated validation task
is retained, so these counts do not measure each dataset's independent quality.
These are unique under the matching rule above, not a guarantee of semantic
independence or unseen database schemas.
