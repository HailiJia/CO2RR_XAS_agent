# November 3 data: model comparison

All 141 configurations / 135 geometry groups retained. Five saved grouped outer folds and three saved inner folds. Three seeds. Sixteen declared candidates: ExtraTrees and Random Forest with leaf sizes 1/2 (64 trees); RBF SVM and logistic regression with C=0.1/1; each tested with signed-shape Combined and DCT16 features. Selection and common threshold fitting use inner predictions only.

Scores below weight each geometry group equally, matching the archived headline metric.

    protocol  micro_f1  macro_f1  exact_set_accuracy
  ExtraTrees  0.784492  0.642650            0.685185
    Logistic  0.683646  0.565539            0.481481
RandomForest  0.749263  0.598730            0.629630
         SVM  0.711916  0.614835            0.566667
    baseline  0.799235  0.662977            0.703704
      nested  0.780448  0.638078            0.677778

Highest individual micro F1: 0.8126036. All three fixed-baseline predictions and headline scores reproduce exactly. No improvement from this search.

The earlier rerun used reconstructed inner folds and record-weighted reporting. Both differed from the archived experiment; the apparent drop was not a clean comparison.

Nested selector minus baseline conditional paired 95% interval: [-0.03409, -0.00536].

Two features and two hyperparameters per family; SVM scores are sigmoid decision margins, uncalibrated. No claim of exhaustive architecture optimization or experimental accuracy.

Manuscript unchanged. Models, selection evidence, held-out predictions, input hashes and model reload checks are saved in this directory.
