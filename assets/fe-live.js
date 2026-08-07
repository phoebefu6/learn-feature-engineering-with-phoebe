/* fe-live.js - the Lumen feature workbench.
   Usage:
     <div class="febox" data-mode="ladder" data-levers="numeric"></div>
     <div class="febox" data-mode="encode"></div>
   data-levers = which feature groups start ON (comma list of:
     numeric,categorical,datetime,interactions,selection,leak).
   "leak" is the trap: it adds a feature derived from the outcome. The TRAIN score
   rockets; press "Score on held-out" and it collapses. That is the whole lesson.

   Honesty rail: the data is synthetic Lumen (deterministic seed 42), but the model
   is real - logistic regression trained by gradient descent in your browser on a
   60/40 split, scored by AUC computed from the ranked held-out predictions.
*/
(function () {
  "use strict";

  /* ---------- deterministic RNG ---------- */

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) {
    var u = 1 - rng(), v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* ---------- the Lumen sessions ---------- */

  var CHANNELS = ["email", "paid_search", "social", "organic", "referral",
                  "display", "affiliate", "direct", "sms"];
  var CHAN_LIFT = { email: 0.8, direct: 0.5, organic: 0.3, referral: 0.2, sms: 0.1,
                    paid_search: -0.1, social: -0.2, affiliate: -0.3, display: -0.5 };
  var CATEGORIES = ["cleanser", "serum", "moisturizer", "set"];
  var DEVICES = ["mobile", "desktop", "tablet"];

  var ROWS = (function () {
    var rng = mulberry32(42);
    var rows = [], i;
    for (i = 0; i < 3000; i++) {
      var returning = rng() < 0.35;
      var channel = CHANNELS[Math.floor(rng() * CHANNELS.length)];
      var device = DEVICES[Math.floor(rng() * DEVICES.length)];
      var category = CATEGORIES[Math.floor(rng() * CATEGORIES.length)];
      var pages = 1 + Math.floor(-Math.log(1 - rng()) * 4);
      var hour = Math.floor(rng() * 24);           // deliberate noise
      var spend = Math.min(400, Math.max(0, 75 * (-Math.log(1 - rng()) * 1.1)));
      var tenureDays = Math.floor(rng() * 900);    // from signup_date
      var weekend = rng() < 0.28;

      /* the true generating process - datetime and interaction effects are REAL,
         so the ladder rungs earn their lift instead of being scripted. */
      var z = -3.05
        + 0.95 * (returning ? 1 : 0)
        + 0.006 * spend
        + 0.11 * pages
        + CHAN_LIFT[channel]
        + 0.9 * (weekend ? 1 : 0)                        // datetime signal
        - 0.0011 * tenureDays                            // datetime signal
        + 0.010 * spend * (returning ? 1 : 0)            // interaction signal
        + 0.35 * (category === "set" ? 1 : 0)
        + 0.25 * gauss(rng);
      var p = 1 / (1 + Math.exp(-z));
      var y = rng() < p ? 1 : 0;

      rows.push({ returning: returning, channel: channel, device: device,
                  category: category, pages: pages, hour: hour, spend: spend,
                  tenureDays: tenureDays, weekend: weekend, y: y,
                  /* the leak: a "feature" recorded after the outcome was known */
                  postPurchaseFlag: y === 1 ? (rng() < 0.93 ? 1 : 0)
                                            : (rng() < 0.05 ? 1 : 0) });
    }
    return rows;
  })();

  /* shuffled split, so train and held-out are exchangeable draws */
  (function () {
    var rng = mulberry32(99), i, j, t;
    for (i = ROWS.length - 1; i > 0; i--) {
      j = Math.floor(rng() * (i + 1));
      t = ROWS[i]; ROWS[i] = ROWS[j]; ROWS[j] = t;
    }
  })();

  var SPLIT = Math.floor(ROWS.length * 0.6);
  var TRAIN = ROWS.slice(0, SPLIT);
  var HOLDOUT = ROWS.slice(SPLIT);

  /* ---------- feature builders (one per lever) ---------- */

  var LEVERS = [
    { key: "numeric",      label: "Raw numerics",  hint: "spend, pages viewed, session hour - straight from the table, unscaled." },
    { key: "categorical",  label: "Encoding",      hint: "one-hot the channel, device and category columns instead of ignoring them." },
    { key: "datetime",     label: "Datetime",      hint: "decompose signup_date into tenure and weekend - time the raw table never exposed." },
    { key: "interactions", label: "Interactions",  hint: "spend x returning, spend-per-page: relationships no single column carries." },
    { key: "selection",    label: "Selection",     hint: "drop the near-useless session_hour noise column." },
    { key: "leak",         label: "Target leakage", hint: "add post_purchase_flag - a column recorded AFTER the outcome. Watch what it does." }
  ];

  /* atPredictionTime = true means "score this row the way production would":
     the leaked column is not yet recorded when the prediction is made, so it
     arrives empty. That absence is what makes leakage a production failure
     rather than a modelling curiosity. */
  function featurize(row, on, atPredictionTime) {
    var f = [1];                                   // intercept
    if (on.numeric) {
      f.push(row.spend / 100, row.pages / 5, (on.selection ? 0 : row.hour / 12));
    }
    if (on.categorical) {
      CHANNELS.forEach(function (c) { f.push(row.channel === c ? 1 : 0); });
      DEVICES.forEach(function (d) { f.push(row.device === d ? 1 : 0); });
      CATEGORIES.forEach(function (c) { f.push(row.category === c ? 1 : 0); });
      f.push(row.returning ? 1 : 0);
    }
    if (on.datetime) {
      f.push(row.tenureDays / 365, row.weekend ? 1 : 0);
    }
    if (on.interactions) {
      f.push((row.spend / 100) * (row.returning ? 1 : 0),
             row.spend / (100 * Math.max(1, row.pages)));
    }
    if (on.leak) {
      f.push(atPredictionTime ? 0 : row.postPurchaseFlag);
    }
    return f;
  }

  /* ---------- real logistic regression (gradient descent) ---------- */

  function train(rows, on) {
    var X = rows.map(function (r) { return featurize(r, on); });
    var y = rows.map(function (r) { return r.y; });
    var d = X[0].length, w = new Array(d).fill(0), i, j, k;
    var lr = 0.5, n = X.length, lambda = 0.001;
    for (k = 0; k < 260; k++) {
      var g = new Array(d).fill(0);
      for (i = 0; i < n; i++) {
        var z = 0;
        for (j = 0; j < d; j++) z += w[j] * X[i][j];
        var p = 1 / (1 + Math.exp(-z));
        var err = p - y[i];
        for (j = 0; j < d; j++) g[j] += err * X[i][j];
      }
      for (j = 0; j < d; j++) w[j] -= lr * (g[j] / n + (j ? lambda * w[j] : 0));
    }
    return w;
  }

  function score(rows, w, on, atPredictionTime) {
    return rows.map(function (r) {
      var f = featurize(r, on, atPredictionTime), z = 0, j;
      for (j = 0; j < f.length; j++) z += w[j] * f[j];
      return { p: 1 / (1 + Math.exp(-z)), y: r.y };
    });
  }

  function auc(scored) {
    var sorted = scored.slice().sort(function (a, b) { return a.p - b.p; });
    var pos = 0, neg = 0, i;
    for (i = 0; i < sorted.length; i++) { if (sorted[i].y) pos++; else neg++; }
    if (!pos || !neg) return 0.5;
    /* rank-sum AUC with average ranks for ties */
    var rankSumPos = 0, i0 = 0;
    while (i0 < sorted.length) {
      var i1 = i0;
      while (i1 + 1 < sorted.length && sorted[i1 + 1].p === sorted[i0].p) i1++;
      var avgRank = (i0 + i1) / 2 + 1;
      for (i = i0; i <= i1; i++) if (sorted[i].y) rankSumPos += avgRank;
      i0 = i1 + 1;
    }
    return (rankSumPos - pos * (pos + 1) / 2) / (pos * neg);
  }

  var CACHE = {};
  function evaluate(on) {
    var key = LEVERS.map(function (l) { return on[l.key] ? 1 : 0; }).join("");
    if (CACHE[key]) return CACHE[key];
    var anyFeature = on.numeric || on.categorical || on.datetime || on.interactions || on.leak;
    var res;
    if (!anyFeature) {
      res = { train: 0.5, holdout: 0.5, nFeatures: 0 };
    } else {
      var w = train(TRAIN, on);
      res = { train: auc(score(TRAIN, w, on, false)),
              holdout: auc(score(HOLDOUT, w, on, true)),
              nFeatures: featurize(TRAIN[0], on, false).length - 1 };
    }
    CACHE[key] = res;
    return res;
  }

  /* ---------- target-encoding demo (mode: encode) ---------- */
  /* A high-cardinality postal_code column that is INDEPENDENT of the target.
     Naive encoding (fit on all training rows, then reuse) leaks; cross-fitted
     encoding (each fold encoded from the other folds) does not. */

  var POSTAL = (function () {
    var rng = mulberry32(7);
    return ROWS.map(function () { return Math.floor(rng() * 1200); });
  })();

  function encodeNaive(idxs) {
    var sum = {}, cnt = {}, i;
    for (i = 0; i < idxs.length; i++) {
      var c = POSTAL[idxs[i]];
      sum[c] = (sum[c] || 0) + ROWS[idxs[i]].y;
      cnt[c] = (cnt[c] || 0) + 1;
    }
    var globalMean = idxs.reduce(function (a, i2) { return a + ROWS[i2].y; }, 0) / idxs.length;
    return function (i2) {
      var c = POSTAL[i2];
      return cnt[c] ? sum[c] / cnt[c] : globalMean;
    };
  }

  function encodeCrossFitted(idxs, folds) {
    var enc = {}, f;
    for (f = 0; f < folds; f++) {
      var others = idxs.filter(function (_, k) { return k % folds !== f; });
      var fn = encodeNaive(others);
      idxs.filter(function (_, k) { return k % folds === f; })
          .forEach(function (i2) { enc[i2] = fn(i2); });
    }
    return enc;
  }

  function encodeDemo() {
    var trainIdx = [], testIdx = [], i;
    for (i = 0; i < ROWS.length; i++) (i < SPLIT ? trainIdx : testIdx).push(i);

    var globalMean = trainIdx.reduce(function (a, k) { return a + ROWS[k].y; }, 0) / trainIdx.length;
    var naiveFn = encodeNaive(trainIdx);
    var crossEnc = encodeCrossFitted(trainIdx, 5);

    function run(getTrainVal) {
      /* single-feature logistic model on the encoded postal code alone */
      var Xt = trainIdx.map(function (k) { return [1, getTrainVal(k) - globalMean]; });
      var yt = trainIdx.map(function (k) { return ROWS[k].y; });
      var w = [0, 0], it, j;
      for (it = 0; it < 300; it++) {
        var g = [0, 0];
        for (j = 0; j < Xt.length; j++) {
          var z = w[0] * Xt[j][0] + w[1] * Xt[j][1];
          var p = 1 / (1 + Math.exp(-z));
          g[0] += (p - yt[j]) * Xt[j][0];
          g[1] += (p - yt[j]) * Xt[j][1];
        }
        w[0] -= 0.5 * g[0] / Xt.length;
        w[1] -= 0.5 * g[1] / Xt.length;
      }
      var trainScored = trainIdx.map(function (k) {
        var v = getTrainVal(k) - globalMean;
        return { p: 1 / (1 + Math.exp(-(w[0] + w[1] * v))), y: ROWS[k].y };
      });
      /* test rows always use the full-training-set encoding, as sklearn does */
      var testScored = testIdx.map(function (k) {
        var v = naiveFn(k) - globalMean;
        return { p: 1 / (1 + Math.exp(-(w[0] + w[1] * v))), y: ROWS[k].y };
      });
      return { train: auc(trainScored), test: auc(testScored) };
    }

    return {
      naive: run(function (k) { return naiveFn(k); }),
      crossFitted: run(function (k) { return crossEnc[k]; })
    };
  }

  /* ---------- rendering ---------- */

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function auc3(x) { return x.toFixed(3); }

  function leverBar(on, onChange) {
    var bar = document.createElement("div");
    bar.className = "ag-levers";
    LEVERS.forEach(function (lv) {
      var chip = document.createElement("button");
      chip.type = "button";
      chip.className = "ag-lever" + (on[lv.key] ? " ag-on" : "") + (lv.key === "leak" ? " fe-trap" : "");
      chip.title = lv.hint;
      chip.textContent = lv.label;
      chip.addEventListener("click", function () {
        on[lv.key] = !on[lv.key];
        chip.classList.toggle("ag-on", on[lv.key]);
        onChange();
      });
      bar.appendChild(chip);
    });
    return bar;
  }

  function rail(text) {
    var p = document.createElement("p");
    p.className = "ag-rail";
    p.textContent = text;
    return p;
  }

  function parseLevers(block) {
    var attr = block.getAttribute("data-levers");
    var start = attr === null ? "numeric,categorical,datetime,interactions,selection" : attr;
    var on = { numeric: false, categorical: false, datetime: false,
               interactions: false, selection: false, leak: false };
    start.split(",").forEach(function (k) {
      k = k.trim(); if (on.hasOwnProperty(k)) on[k] = true;
    });
    return on;
  }

  /* ---- ladder mode ---- */

  function wireLadder(block) {
    block.classList.add("febox-ready");
    var on = parseLevers(block);
    var revealed = false;

    var bar = document.createElement("div");
    bar.className = "sql-bar";
    bar.innerHTML = '<span class="sql-dot"></span><span class="sql-title">Lumen conversion model - feature workbench</span>';
    block.appendChild(bar);

    block.appendChild(leverBar(on, function () { revealed = false; render(); }));

    var ctrl = document.createElement("div");
    ctrl.className = "ev-ctrl";
    ctrl.innerHTML = '<button type="button" class="sql-btn sql-run fe-holdout">Score on held-out</button>' +
                     '<span class="ev-lab fe-note">Training score is what you see while you work. Held-out is what ships.</span>';
    block.appendChild(ctrl);

    var tiles = document.createElement("div"); tiles.className = "ev-tiles";
    block.appendChild(tiles);
    var verdict = document.createElement("div");
    block.appendChild(verdict);
    block.appendChild(rail("Real model, real split: logistic regression trained by gradient descent on 1,800 Lumen sessions, scored by AUC on 1,200 held-out sessions your model never saw. Only the data is synthetic."));

    function render() {
      var r = evaluate(on);
      var gap = r.train - r.holdout;
      tiles.innerHTML =
        '<div class="ev-tile"><b>' + auc3(r.train) + '</b><span>training AUC</span></div>' +
        '<div class="ev-tile ' + (revealed ? "ev-money" : "fe-hidden") + '"><b>' +
          (revealed ? auc3(r.holdout) : "? ? ?") + '</b><span>held-out AUC</span></div>' +
        '<div class="ev-tile"><b>' + r.nFeatures + '</b><span>features built</span></div>' +
        '<div class="ev-tile"><b>' + (revealed ? (gap >= 0 ? "+" : "") + gap.toFixed(3) : "-") + '</b><span>train minus held-out</span></div>';
      if (!revealed) {
        verdict.className = "ag-verdict ag-quiet";
        verdict.textContent = "";
        return;
      }
      if (on.leak) {
        verdict.className = "ag-verdict ag-fail";
        verdict.textContent = "LEAKAGE - training AUC " + auc3(r.train) + " looked like the best model you have ever built. Held-out: " + auc3(r.holdout) +
          ", a gap of " + gap.toFixed(3) + ". post_purchase_flag is recorded AFTER the sale, so it will be empty at prediction time. sklearn's own noise experiment shows the same shape: a leaked selection step scored 0.76 accuracy on data with NO signal, where the honest score was 0.50.";
      } else if (r.nFeatures === 0) {
        verdict.className = "ag-verdict ag-fail";
        verdict.textContent = "No features, no model - AUC 0.500 is a coin flip. Every point above this line is something you build, not something the algorithm finds.";
      } else {
        verdict.className = "ag-verdict ag-pass";
        verdict.textContent = "Honest score: held-out AUC " + auc3(r.holdout) + " from " + r.nFeatures +
          " features, with a train-holdout gap of " + gap.toFixed(3) + ". Same model class throughout - every point of lift came from the columns you built.";
      }
    }

    ctrl.querySelector(".fe-holdout").addEventListener("click", function () {
      revealed = true; render();
    });
    render();
  }

  /* ---- encode mode (target-encoding cross-fitting) ---- */

  function wireEncode(block) {
    block.classList.add("febox-ready");
    var res = encodeDemo();

    var bar = document.createElement("div");
    bar.className = "sql-bar";
    bar.innerHTML = '<span class="sql-dot"></span><span class="sql-title">postal_code (1,200 levels, independent of the target) - target encoding two ways</span>';
    block.appendChild(bar);

    var table = document.createElement("div");
    table.className = "ag-score-table";
    function row(name, r, ok, note) {
      var d = document.createElement("div");
      d.className = "ag-score-row " + (ok ? "ag-row-pass" : "ag-row-fail");
      d.innerHTML = '<span class="ag-mark">' + (ok ? "✓" : "✗") + "</span>" +
        '<span class="ag-q">' + esc(name) + '<span class="ag-why">' + esc(note) + "</span></span>" +
        '<span class="ag-out">train AUC ' + auc3(r.train) + "  ·  held-out AUC " + auc3(r.test) + "</span>";
      return d;
    }
    table.appendChild(row("Naive encoding - fit on all training rows, then reuse", res.naive, false,
      "each row's own outcome helped build its own encoding"));
    table.appendChild(row("Cross-fitted encoding - each fold encoded from the other folds", res.crossFitted, true,
      "sklearn's TargetEncoder.fit_transform does exactly this"));
    block.appendChild(table);

    var verdict = document.createElement("div");
    verdict.className = "ag-verdict ag-fail";
    verdict.textContent = "The postal code is RANDOM - it knows nothing about who converts. Yet naive encoding scored " +
      auc3(res.naive.train) + " on training data and collapsed to " + auc3(res.naive.test) +
      " on held-out. Cross-fitting keeps train and held-out honest (" + auc3(res.crossFitted.train) + " / " + auc3(res.crossFitted.test) +
      "). sklearn's own demo shows the same failure on 50,000 rows: test R2 0.634 without cross-fitting versus 0.793 with it - their docs call it catastrophic overfitting.";
    block.appendChild(verdict);
    block.appendChild(rail("Both encoders are implemented here and run in your browser on the same 1,800 training / 1,200 held-out split. The postal codes are drawn independently of the outcome, so any apparent signal is leakage by construction."));
  }

  /* ---------- boot ---------- */

  function boot() {
    var blocks = document.querySelectorAll(".febox");
    Array.prototype.forEach.call(blocks, function (block) {
      if (block.classList.contains("febox-ready")) return;
      var mode = block.getAttribute("data-mode") || "ladder";
      if (mode === "encode") wireEncode(block); else wireLadder(block);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else { boot(); }
})();
