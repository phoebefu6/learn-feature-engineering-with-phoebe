# Official course map - learn-feature-engineering-with-phoebe

Research date: 2026-08-06. All docs, guides, and papers fetched live and verified. Version-sensitive: sklearn renumbered its user-guide chapters (preprocessing is now 8.3, compose 8.1, pitfalls ch.12) - anchor by section NAME, never number, on the pages.

## Positioning vs siblings (state honestly on pages)
- **learn-intro-ml** b9 does `ColumnTransformer` + `Pipeline` basics and exposes `session_hour` as noise. It explicitly defers "heavy feature engineering" here.
- **learn-classification-regression** teases feature engineering and defers here.
- **learn-model-evaluation** judges the result (metrics, thresholds, cost). This course owns the INPUTS: making columns a model can learn from, and the traps that fake success.
- **learn-timeseries-forecasting** owns lag/rolling features. **learn-deep-learning** owns representation learning. Named, not taught.

## Sources

| # | Source | URL | Depth |
|---|--------|-----|-------|
| S1 | sklearn "Preprocessing data" user guide (8.3) | scikit-learn.org/stable/modules/preprocessing.html | full chapter |
| S2 | sklearn "Common pitfalls and recommended practices" (ch.12, data leakage) | scikit-learn.org/stable/common_pitfalls.html | full chapter |
| S3 | sklearn "Pipelines and composite estimators" (8.1) | scikit-learn.org/stable/modules/compose.html | full chapter |
| S4 | sklearn "Feature selection" (1.13) | scikit-learn.org/stable/modules/feature_selection.html | full chapter |
| S5 | sklearn "Permutation feature importance" (5.2) | scikit-learn.org/stable/modules/permutation_importance.html | full page |
| S6 | sklearn `TargetEncoder` + "Target Encoder's Internal Cross fitting" example | scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html · /auto_examples/preprocessing/plot_target_encoder_cross_val.html | class + example |
| S7 | Kaggle Learn "Feature Engineering" (Ryan Holbrook) | kaggle.com/learn/feature-engineering | 7 lessons, ~5h |
| S8 | Google MLCC "Working with numerical data" | developers.google.com/machine-learning/crash-course/numerical-data | 85 min, 11 lessons |
| S9 | Google MLCC "Working with categorical data" | developers.google.com/machine-learning/crash-course/categorical-data | 50 min, 6 lessons |
| S10 | Google MLCC "Embeddings" + glossary | developers.google.com/machine-learning/crash-course/embeddings · /glossary | 45 min |
| S11 | Google "Rules of Machine Learning" (Zinkevich) - rules 16-22, 29 | developers.google.com/machine-learning/guides/rules-of-ml | doc |
| S12 | Feast feature-store docs (concepts, point-in-time joins, feature retrieval) | docs.feast.dev | doc set |
| S13 | OpenAI Embeddings guide + Sentence Transformers | developers.openai.com/api/docs/guides/embeddings · sbert.net | 2 doc sets |
| S14 | Pargent, Pfisterer, Thomas, Bischl (2021) "Regularized target encoding outperforms traditional methods..." | arxiv.org/abs/2104.00629 | paper |

## Verified key facts (build against these; quote numbers exactly)

### Leakage - the course spine (S2, sklearn's own words + numbers)
- **Definition (verbatim):** "Data leakage occurs when information that would not be available at prediction time is used when building the model. This results in overly optimistic performance estimates... and thus poorer performance when the model is used on actually novel data."
- **The three avoidance rules (verbatim):** (1) "Always split the data into train and test subsets first, particularly before any preprocessing steps." (2) "Never include test data when using the `fit` and `fit_transform` methods." - use `fit_transform` on train, `transform` on test. (3) "The scikit-learn pipeline is a great way to prevent data leakage as it ensures that the appropriate method is performed on the correct data subset."
- **THE WORKED EXAMPLE (random noise data, `SelectKBest(k=25)` + `HistGradientBoostingClassifier`):** selection fit on ALL data then split -> **accuracy 0.76**; split first, fit on train only -> **0.50**; pipeline -> **0.50**; pipeline + `cross_val_score` -> **0.43 +/- 0.05**. Leakage manufactured 26 accuracy points out of PURE NOISE.
- **Pipeline's stated safety purpose (S3, verbatim):** "Pipelines help avoid leaking statistics from your test data into the trained model in cross-validation, by ensuring that the same samples are used to train the transformers and predictors."

### Target encoding (S6, S14) - the subtle leak that survives a naive split
- Formula (binary): `S_i = λ_i * (n_iY / n_i) + (1 - λ_i) * (n_Y / n)`, `λ_i = n_i / (m + n_i)`, m = smoothing. `smooth="auto"` uses empirical Bayes `m = σ_i² / τ²`. "Large smoothing factors put more weight on the global mean."
- **Cross-fitting (verbatim):** "In `fit_transform`, the training data is split into k folds (determined by the `cv` parameter) and each fold is encoded using the encodings learnt using the other k-1 folds." Purpose: "to prevent target information from leaking into the train-time representation, especially for non-informative high-cardinality categorical variables."
- **`fit(X, y).transform(X)` does NOT equal `fit_transform(X, y)`** - the docs explicitly discourage `fit` then `transform` on training data.
- **THE DEMO NUMBERS (50,000 rows; `near_unique` column ~45,000 distinct values, independent of target):** WITH cross-fitting train R² **0.800** / test **0.793**; WITHOUT cross-fitting train **0.858** / test **0.634**. Docs call the no-cross-fitting case "catastrophic overfitting."
- Use when: "categorical features with high cardinality, where one-hot encoding would inflate the feature space" - classic example zip code / region.
- `shuffle` and `random_state` **deprecated since sklearn 1.9** - do not teach them.
- S14 (peer-reviewed): "regularized versions of target encoding... consistently provided the best results" vs integer and one-hot encoding on high-cardinality features.

### Numeric transforms (S1, S8)
- **Google's when-to-use table (teach this shape):** linear/min-max scaling -> "feature is mostly uniformly distributed" (flat); z-score -> "normally distributed (peak close to mean)" (bell); log -> "heavy skewed on at least either side of tail"; clipping -> "contains extreme outliers". Worked: range 100-900, x=300 -> 0.25. z-score μ=100 σ=20 x=130 -> +1.5.
- Why normalize (S8, verbatim list): converge more quickly · infer better predictions · avoid the "NaN trap" · learn appropriate weights per feature.
- **sklearn's scaler guidance:** outliers -> `RobustScaler` ("drop-in replacement", "more robust estimates for the center and range"); sparse data -> `MaxAbsScaler` ("specifically designed for scaling sparse data, and is the recommended way"); `StandardScaler` on sparse needs `with_mean=False` or raises ValueError ("silently centering would break the sparsity").
- `PowerTransformer`: `yeo-johnson` or `box-cox`; **"Box-Cox can only be applied to strictly positive data."** Applies zero-mean unit-variance to output by default.
- **Binning (S8):** "groups different numerical subranges into bins or buckets"; bin when "the overall linear relationship between the feature and the label is weak or nonexistent" OR "feature values are clustered". "The model learns separate weights for each bin." Quantile bucketing "gives extra information space to the large torso while compacting the long tail into a single bucket."
- `KBinsDiscretizer` strategies: `uniform` / `quantile` ("equally populated bins") / `kmeans`. Accepts per-feature `n_bins=[3,2,2]`.
- `SplineTransformer` vs `PolynomialFeatures`: splines avoid "oscillatory behaviour at the boundaries... known as Runge's phenomenon", extrapolate better, banded matrix with low condition number (vs Vandermonde). **"The SplineTransformer treats each feature separately, i.e. it won't give you interaction terms."** Equivalence: `degree=0` spline == `KBinsDiscretizer(encode='onehot-dense')` with `n_bins = n_knots - 1`.
- Polynomial transforms (S8): synthetic x2 = x1² lets a linear model draw a curve. Caution: "If you transform a feature in a way that changes its scale, you should consider experimenting with normalizing it as well."

### Categorical (S1, S9)
- **Why integer-encoding is wrong (S9, verbatim):** "the model would treat the indexed values as continuous floating-point numbers. The model would then consider 'purple' six times more likely than 'orange.'"
- `OneHotEncoder`: `handle_unknown` = error/ignore/`infrequent_if_exist`; `min_frequency` (int or fraction) and `max_categories` group rare levels into an infrequent bucket ("`max_categories` includes the feature that combines infrequent categories"); `drop='first'|'if_binary'` -> n-1 columns, "useful to avoid co-linearity". `np.nan` and `None` are SEPARATE categories.
- High dimensionality (S9, verbatim): "Categorical data tends to produce high-dimensional feature vectors... High dimensionality increases training costs and makes training more difficult." "For natural-language data, the main method of reducing dimensionality is to convert feature vectors to embedding vectors."
- **Feature crosses (S9):** "created by crossing (taking the Cartesian product of) two or more categorical or bucketed features". Leaf example: 3 edges x 2 arrangements -> 6 combos, lobed+alternate -> {0,0,0,0,0,1}. **Warning: 100-element x 200-element cross -> 20,000-element sparse feature.**

### Selection + importance (S4, S5)
- Classes: `VarianceThreshold` (Bernoulli threshold example: remove features that are one value in >80% of samples -> `threshold = .8*(1-.8) = 0.16`), `SelectKBest`/`SelectPercentile`, `chi2`/`f_classif`/`mutual_info_classif` (classification) vs `f_regression`/`mutual_info_regression`, `RFE`/`RFECV`, `SelectFromModel` (L1-based: smaller C / larger alpha -> fewer features; tree-based), `SequentialFeatureSelector`.
- **Warning (verbatim):** "Beware not to use a regression scoring function with a classification problem, you will get useless results."
- SFS vs RFE: "SFS differs from RFE and SelectFromModel in that it does not require the underlying model to expose a `coef_` or `feature_importances_` attribute. It may however be slower."
- **Impurity-importance caveat (S5, verbatim):** impurity-based importance for trees is "strongly biased" and "favor high cardinality features (typically numerical features) over low cardinality features such as binary features or categorical variables with a small number of possible categories"; "can give high importance to features that may not be predictive on unseen data when the model is overfitting." Permutation importance "avoids this issue, since it can be computed on unseen data."
- **Held-out rule (verbatim):** "Features that are important on the training set but not on the held-out set might cause the model to overfit."
- **The meta-warning (verbatim):** "Features that are deemed of low importance for a bad model (low cross-validation score) could be very important for a good model... Permutation importance does not reflect the intrinsic predictive value of a feature by itself but how important this feature is for a particular model."
- Correlated-feature caveat: permuting one leaves the model access via its correlate, so BOTH report low; fix by clustering correlated features and keeping one per cluster.
- NOTE: the "select inside CV" leakage warning lives in S2 (common pitfalls), NOT in the feature-selection page - cite correctly.

### Mutual information (S7, Kaggle lesson 2)
- "a measure of the extent to which knowledge of one quantity reduces uncertainty about the other." Beats correlation because "correlation only detects linear relationships."
- Scale: min 0.0 (independence); "values above 2.0 or so are uncommon."
- **Three caveats (verbatim):** "MI can't detect interactions between features. It is a univariate metric." · "The actual usefulness of a feature depends on the model you use it with." · "Just because a feature has a high MI score doesn't mean your model will be able to do anything with that information."
- Kaggle lesson list (7): What Is Feature Engineering · Mutual Information · Creating Features · Clustering With K-Means · Principal Component Analysis · Target Encoding · Feature Engineering for House Prices.

### Governance + stores (S11, S12)
- **Training-serving skew (S11, verbatim - Feast docs do NOT use this term, cite Google):** "a difference between performance during training and performance during serving," caused by "a discrepancy between how you handle data in the training and serving pipelines," "a change in the data between when you train and when you serve," or "a feedback loop between your model and your algorithm."
- **Rule #29 (the doctrinal case for a feature store):** "The best way to make sure that you train like you serve is to save the set of features used at serving time, and then pipe those features to a log to use them at training time."
- Rules 16-22: #17 "Start with directly observed and reported features as opposed to learned features." · #19 "Use very specific features when you can." · #20 "Combine and modify existing features to create new features in human-understandable ways." · #21 "The number of feature weights you can learn in a linear model is roughly proportional to the amount of data you have." · #22 "Clean up features you are no longer using."
- **Feast (S12):** offline store (historical extraction for training) + online store (low-latency serving). Solves: consistent availability train+serve, **"Avoid data leakage by generating point-in-time correct feature sets"**, single data-access layer. Point-in-time joins scan backward from each entity-dataframe timestamp up to a TTL ("TTL is not relative to the current point in time"). Feast is NOT an ETL system, orchestrator, warehouse, or database; does NOT fully solve lineage, drift detection, or batch feature engineering.

### Embeddings bridge (S10, S13)
- One-hot problem (S10, verbatim): with M one-hot entries and N first-layer nodes "the model has to train MxN weights for that layer"; one-hot encodings "lack meaningful relationships" - "hot dogs and shawarmas" should sit closer than "hot dogs and salads."
- Glossary: embedding vector = "A relatively low-dimensional vector of floating-point values (typically, between 5 and 500)".
- **THE BRIDGE QUOTE (S13, OpenAI, verbatim):** "An embedding can be used as a general free-text feature encoder within a machine learning model."
- Dimensions/cost: `text-embedding-3-small` 1536 dims (62,500 pages/$), `text-embedding-3-large` 3072 (9,615 pages/$). The `dimensions` parameter lets you "trade-off performance and cost"; a 3-large embedding "shortened to a size of 256" still outperforms an unshortened ada-002 at 1536.
- Local option: Sentence Transformers `all-MiniLM-L6-v2` -> `[3, 384]` dims, 10,000+ pretrained models.
- Rule #17 (S11) is the honest counterweight: start with observed/reported features BEFORE learned ones.

## Running case - Lumen (reuse the intro-ml canon exactly, do not re-derive)

Lumen Skincare, $18M/yr DTC, 40,000 checkout sessions, `converted` base rate ~3.2%, `order_value` mean ~$74 right-skewed. Features: `prior_30d_spend` (strongest, r~0.6 with order_value), `new_vs_returning`, `channel` (9), `product_category` (4), `device`, `pages_viewed`, `session_hour` (deliberate near-useless noise - intro-ml exposed it; b8 here retires it properly).

**This course adds three columns to the case (course-specific, state as such):** `postal_code` (high-cardinality ~1,200 levels - the target-encoding lesson), `signup_date` (datetime decomposition), `support_note` (free text - the b9 embeddings bridge). All synthetic, same seed discipline.

## fe-live.js canon (hard-coded rungs - verify in-browser BEFORE fan-out, then quote exactly)

A real in-browser logistic scorer over embedded Lumen features. Levers add feature groups; the score shown is **held-out AUC**.

VERIFIED IN-BROWSER 2026-08-06. 3,000 sessions, shuffled 60/40 split (1,800 train / 1,200 held-out), logistic regression by gradient descent. **Quote these exactly.**

| Rung | Levers on | Train AUC | Held-out AUC |
|------|-----------|-----------|--------------|
| 0 | nothing (no features) | 0.500 | **0.500** |
| 1 | raw numerics | 0.676 | **0.696** |
| 2 | + encoding | 0.782 | **0.785** |
| 3 | + datetime | 0.798 | **0.806** |
| 4 | + interactions | 0.806 | **0.812** |
| 5 | + selection (drops session_hour) | 0.806 | **0.813** |

**The trap (anti-lever #4):** the `target leakage` lever adds `post_purchase_flag`, recorded AFTER the sale. Held-out rows are scored **the way production would score them** - that column arrives EMPTY, because it does not exist yet at prediction time. Result: **train 0.967, held-out 0.789** (gap **0.179**) - and 0.789 is BELOW the honest 0.813, so the leak made the model worse than never adding it. Teach alongside sklearn's own documented result: a leaked selection step scored **0.76 accuracy on pure-noise data** where the honest score was **0.50** (0.43 under CV).

**The encode-mode demo (b3):** `postal_code` with 1,200 levels drawn INDEPENDENTLY of the target. Naive encoding (fit on all training rows, reuse) -> **train AUC 0.931, held-out 0.493**. Cross-fitted (each fold encoded from the other folds, what `TargetEncoder.fit_transform` does) -> **train 0.509, held-out 0.493**. A column that knows nothing scored 0.931 in training. Pair with sklearn's 50,000-row demo: test R² **0.634** without cross-fitting vs **0.793** with it.

Embeds: `<div class="febox" data-mode="ladder" data-levers="numeric"></div>` (empty data-levers = all off; absent attribute = the five honest levers on, leak off) · `<div class="febox" data-mode="encode"></div>`.

Honesty rail: the model, the split, and every AUC are computed live in-browser from embedded Lumen data; only the data is synthetic. Numbers quoted from sklearn/Google docs are cited on the page.

## Per-session coverage - leader track (6 x 45 min)

| Session | Covers | S2 | S5 | S6 | S8/S9 | S11 | S12 | S13 |
|---------|--------|----|----|----|-------|-----|-----|-----|
| a1 Why features decide ROI before models | data work dominates ML time; quality > quantity; where effort pays | ◐ | | | ✓ | ◐ | | |
| a2 The leakage disaster, in plain English | the 0.76-on-noise result; why a suspiciously good test score is a red flag | ✓ | ◐ | ◐ | | | | |
| a3 Data quality is a feature budget | scrubbing categories, missingness, cardinality, drift-prone columns | | | | ✓ | ✓ | | |
| a4 Feature governance + feature stores | training-serving skew, Rule #29, point-in-time correctness, when a store pays | | | | | ✓ | ✓ | |
| a5 Hand-crafted vs learned features | embeddings for decision-makers; Rule #17 counterweight; cost tradeoffs | | | | ◐ | ✓ | | ✓ |
| a6 Questions to ask your DS team | the feature review checklist + what a feature spec must contain | ✓ | ✓ | ◐ | | ✓ | ◐ | |

## Per-session coverage - practitioner track (10 x 45 min)

| Session | Covers | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S13 |
|---------|--------|----|----|----|----|----|----|----|----|----|-----|-----|-----|
| b1 What a feature actually is | raw column vs feature, the model's-eye view, fe-live rung 0 | ◐ | | | | | | ✓ | ✓ | ◐ | | ◐ | |
| b2 Numeric transforms | scaler family + when-to-use table, skew/log/clip, binning, splines vs polys | ✓ | | | | | | | ✓ | | | | |
| b3 Categorical encoding | one-hot params, rare grouping, high cardinality, target encoding + cross-fitting | ✓ | ◐ | ◐ | | | ✓ | ◐ | | ✓ | | | |
| b4 Datetime + cyclical | decomposition, cyclical encoding, recency/tenure, session_hour revisited | ◐ | | | | | | ◐ | ◐ | | | | |
| b5 Leakage: the session | the 4 families, sklearn's 0.76-vs-0.50, fit-on-train-only, Pipeline as structure | ◐ | ✓ | ✓ | ◐ | | ◐ | | | | | | |
| b6 Interactions + aggregations | ratios, feature crosses (20,000-element warning), group aggregates, entity history | ✓ | | | | | | ✓ | ◐ | ✓ | | ◐ | |
| b7 Missing data as signal | imputation strategies, missingness indicators, NaN-as-category | ✓ | ◐ | ◐ | | | | | ✓ | ◐ | | | |
| b8 Selection + importance | variance/univariate/RFE/model-based, MI caveats, permutation vs impurity | | ◐ | ◐ | ✓ | ✓ | | ✓ | | | | ◐ | |
| b9 Embeddings as features | text column -> embedding -> tabular model; dims/cost; when to stop hand-crafting | | | ◐ | | | | | | ◐ | ✓ | ◐ | ✓ |
| b10 Capstone: the feature spec | full pipeline on Lumen + written spec (definition, source, leakage check, owner) | ✓ | ✓ | ✓ | ◐ | ◐ | ◐ | ◐ | | | | ✓ | ◐ |

✓ = ~80% of that source's working content for the topic. ◐ = partial/contextual. Certificates and graded exercises stay with the official providers (Kaggle Learn, Google MLCC) - say so on the pages.

## Overlap analysis
Taught ONCE: the train/test discipline (b5, referenced everywhere after), the Pipeline pattern (b5, reused in b10), encoding basics (b3). Deltas: S8-only (the scaling when-to-use table, binning criteria) -> b2. S6/S14-only (target encoding math + cross-fitting) -> b3. S2-only (the noise experiment) -> b5/a2. S9-only (feature crosses + dimensionality warning) -> b6. S5-only (permutation vs impurity) -> b8. S13-only (embeddings as a feature encoder) -> b9. S11/S12-only (skew, Rule #29, point-in-time) -> a4/b10.

## Open lane (differentiation)
A leakage trap you can PRESS (train score rockets, held-out button collapses it) backed by sklearn's own 0.76-on-noise number · target encoding taught with its cross-fitting math AND its failure numbers, not just "use TargetEncoder" · one running dataset from raw columns to written feature spec · classic tabular craft with an honest embeddings bridge instead of pretending either replaces the other.

## Not covered by design (honest list)
- Deep-learning representation learning -> learn-deep-learning-with-phoebe
- Lag/rolling/seasonal time-series features -> learn-timeseries-forecasting-with-phoebe
- Feature-store implementation and ops (we teach the concept + when it pays) -> deng bucket
- Automated feature engineering tools (featuretools, autoML) - named, not taught
- Image/audio feature extraction - out of a tabular course's scope
- Certificates and graded exercises stay with Kaggle Learn and Google MLCC
