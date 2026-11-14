# Evaluation correction — fixed top-two protocol

The v3 result 0.7821/0.6448/exact-set0.6778 re-ran automatic feature/**depth**/threshold selection. It is a historical automatic-depth result and **does not update the fixed top-two-layer manuscript benchmark**. Its saved models, arrays and protocol are retained unchanged as provenance. The corrected paired experiments are in `../top_two_20261023`, using only population-weighted layers 5+4; the author selected Combined spectrum/CDF/derivatives as the fixed-feature anchor.

The three added independent configurations are all CO-only. Twelve extra layer spectra do not create twelve independent samples. CHO/OCHO/COH and mixture-positive supports did not increase.

For the three-case v3 new-only holdout, report **CO precision 1.0, recall 2/3, F1 0.8**, and exact-set 2/3. The other ten labels have no positive cases, so their recovery was not tested. The saved eleven-label macro F1 0.0727 is 0.8/11 because absent labels are assigned zero; it is provenance, not an informative headline for this subset. Do not interpret it as evidence of poor recovery of species that were absent.

Representation changed in all five v3 folds. On the old 87 configurations, four previously incorrect identity sets became correct and five previously correct sets became wrong; two CO-only cases gained false H/OH. These tradeoffs accompany CO mixture detection changing 5/9→7/9. Exact case tables and verified metric context are in the corrected study.
