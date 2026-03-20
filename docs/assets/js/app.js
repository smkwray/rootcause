/* ============================================================
   Rootcause — Site Renderer
   Fetches site_data.json and populates the page.
   ============================================================ */

(function () {
  "use strict";

  /* --- Theme toggle --- */
  var toggle = document.getElementById("theme-toggle");
  var html = document.documentElement;

  function setTheme(theme) {
    html.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("rootcause-theme", theme);
    } catch (_) {}
    if (toggle) {
      toggle.innerHTML =
        theme === "dark"
          ? '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>'
          : '<svg viewBox="0 0 24 24"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>';
    }
  }

  var saved;
  try {
    saved = localStorage.getItem("rootcause-theme");
  } catch (_) {}
  if (saved) {
    setTheme(saved);
  } else if (
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  ) {
    setTheme("dark");
  } else {
    setTheme("light");
  }

  if (toggle) {
    toggle.addEventListener("click", function () {
      var current = html.getAttribute("data-theme");
      setTheme(current === "dark" ? "light" : "dark");
    });
  }

  if (window.matchMedia) {
    window
      .matchMedia("(prefers-color-scheme: dark)")
      .addEventListener("change", function (e) {
        try {
          if (!localStorage.getItem("rootcause-theme")) {
            setTheme(e.matches ? "dark" : "light");
          }
        } catch (_) {
          setTheme(e.matches ? "dark" : "light");
        }
      });
  }

  /* --- Lookup tables (must be assigned before render is called) --- */
  var CHECK_NAMES = {
    event_pretrend: "Pre-trends",
    overlap: "Overlap",
    support_trim: "Support trim",
    border_design: "Border design",
    negative_control_outcomes: "Neg. controls",
    staggered_att: "Staggered ATT",
    min_wage_redesign: "MW redesign",
  };

  var LANE_SUMMARIES = {
    min_wage_property:
      "TWFE and DML disagree on direction: the conventional fixed-effects estimate is positive but insignificant, while DML with richer controls is negative and marginal. Border-county first-difference designs do not confirm either direction. Negative-control outcomes show significant associations with slow-moving demographics, raising concern about residual confounding.",
    min_wage_violent:
      "Both TWFE and DML point positive, but the conventional estimate is far from significant. The DML result is marginally significant and survives support trimming. However, the staggered-adoption design is not interpretable in this sample, and negative-control outcome tests flag potential confounding.",
    eitc_property:
      "Pre-trends pass, but neither TWFE nor DML detects a statistically significant effect. The support-trimmed DML estimate shifts substantially, and one negative-control outcome flags. Results are method-sensitive and should not be treated as confirmatory.",
    eitc_violent:
      "Pre-trends pass and DML is marginally significant, but the TWFE estimate is weak. Support trimming attenuates the DML signal. Border-county designs show no significant effect. One negative-control outcome flags. This lane is suggestive but not robust.",
    snap_bbce_property:
      "Event-study pre-trends fail, indicating the parallel-trends assumption is unlikely to hold. The staggered-adoption pre-trend also fails. This lane is not suitable for causal claims in its current form.",
    snap_bbce_violent:
      "Event-study pre-trends fail. While the staggered-adoption pre-trend passes, the conventional event-study failure means this lane remains exploratory only.",
    tanf_property:
      "Covariate overlap is poor (max SMD exceeds 1.0), meaning treated and control groups differ substantially on observables. Neither TWFE nor DML detects a clear signal. This lane lacks the statistical support for causal interpretation.",
    tanf_violent:
      "Same overlap concerns as the property lane. No estimation method produces a clear signal. This lane should be treated as exploratory pending better identification.",
  };

  var BIDIR_UNITS = {
    poverty_to_violent: "A 1 percentage-point increase in the poverty rate is associated with this change in the violent crime rate (per 100,000 residents).",
    poverty_to_property: "A 1 percentage-point increase in the poverty rate is associated with this change in the property crime rate (per 100,000 residents).",
    violent_to_poverty: "A 1-unit increase in the violent crime rate (per 100,000) is associated with this change in the poverty rate (percentage points). Coefficients are very small because the scales differ.",
    property_to_poverty: "A 1-unit increase in the property crime rate (per 100,000) is associated with this change in the poverty rate (percentage points). Coefficients are very small because the scales differ.",
  };

  /* --- Data loading --- */
  if (window.SITE_DATA) {
    try {
      render(window.SITE_DATA);
    } catch (e) {
      var errEl = document.getElementById("data-content");
      if (errEl) errEl.innerHTML = '<p style="color:red;padding:2rem;">Render error: ' + e.message + '</p>';
    }
  }

  /* --- Render --- */
  function render(data) {
    renderPanelStats(data.panel);
    renderLanes(data.lanes);
    renderBidirectional(data.bidirectional);
    renderSources(data.sources);

    var loaders = document.querySelectorAll(".loading");
    for (var i = 0; i < loaders.length; i++) {
      loaders[i].style.display = "none";
    }
  }

  /* --- Panel stats (hero) --- */
  function renderPanelStats(panel) {
    setText("stat-counties", panel.counties.toLocaleString());
    setText("stat-years", panel.year_min + "\u2013" + panel.year_max);
    setText("stat-obs", panel.rows.toLocaleString());
  }

  /* --- Lane cards --- */
  function renderLanes(lanes) {
    var primary = [];
    var secondary = [];
    var exploratory = [];

    lanes.forEach(function (lane) {
      if (lane.tier === "primary") primary.push(lane);
      else if (lane.tier === "secondary") secondary.push(lane);
      else exploratory.push(lane);
    });

    renderLaneGroup("primary-lanes", primary);
    renderLaneGroup("secondary-lanes", secondary);
    renderLaneGroup("exploratory-lanes", exploratory);
  }

  function renderLaneGroup(containerId, lanes) {
    var container = document.getElementById(containerId);
    if (!container || lanes.length === 0) return;

    var out = "";
    lanes.forEach(function (lane) {
      out += buildLaneCard(lane);
    });
    container.innerHTML = out;
  }

  function buildLaneCard(lane) {
    var cred = lane.credibility || {};
    var statusDesc = cred.status_description || "";
    var checks = cred.checks || [];

    var tierClass =
      lane.tier === "primary"
        ? "badge-primary"
        : lane.tier === "secondary"
          ? "badge-secondary"
          : "badge-exploratory";

    var h = '<div class="lane-card">';
    h += '<div class="lane-header">';
    h += '<div class="lane-title">' + esc(formatLaneTitle(lane.title)) + "</div>";
    h += '<div class="lane-badges">';
    h +=
      '<span class="badge ' +
      tierClass +
      '">' +
      esc(lane.tier_label) +
      "</span>";
    if (statusDesc) {
      h += '<span class="badge badge-caution">' + esc(statusDesc) + "</span>";
    }
    h += "</div></div>";

    h += '<table class="estimates-table">';
    h += "<thead><tr>";
    h += "<th>Method</th><th>Estimate</th><th>p-value</th><th>95% CI</th>";
    h += "</tr></thead><tbody>";

    if (lane.twfe) {
      h += buildEstimateRow(
        "TWFE",
        lane.twfe.coefficient,
        lane.twfe.p_value,
        lane.twfe.ci_lower,
        lane.twfe.ci_upper
      );
    }
    if (lane.dml) {
      h += buildEstimateRow(
        "DML",
        lane.dml.theta,
        lane.dml.p_value,
        lane.dml.ci_lower,
        lane.dml.ci_upper
      );
    }
    h += "</tbody></table>";

    if (checks.length > 0) {
      h += '<div class="checks-row">';
      checks.forEach(function (c) {
        h += buildCheckPill(c);
      });
      h += "</div>";
    }

    var summary = getLaneSummary(lane);
    if (summary) {
      h += '<div class="lane-summary">' + esc(summary) + "</div>";
    }

    h += "</div>";
    return h;
  }

  function buildEstimateRow(method, value, pval, ciLow, ciHigh) {
    var pClass =
      pval < 0.05 ? "p-sig" : pval < 0.1 ? "p-marginal" : "p-nonsig";
    var h = "<tr>";
    h += '<td class="est-method">' + method + "</td>";
    h += '<td class="est-value">' + formatNum(value) + "</td>";
    h += '<td class="est-value ' + pClass + '">' + formatP(pval) + "</td>";
    h +=
      '<td class="est-ci">[' +
      formatNum(ciLow) +
      ", " +
      formatNum(ciHigh) +
      "]</td>";
    h += "</tr>";
    return h;
  }

  function buildCheckPill(check) {
    var statusClass =
      check.status === "pass"
        ? "check-pass"
        : check.status === "warn"
          ? "check-warn"
          : check.status === "fail"
            ? "check-fail"
            : "check-na";
    return (
      '<span class="check-pill ' +
      statusClass +
      '" title="' +
      esc(check.detail) +
      '"><span class="dot"></span>' +
      esc(formatCheckName(check.name)) +
      "</span>"
    );
  }

  function formatCheckName(name) {
    return CHECK_NAMES[name] || name.replace(/_/g, " ");
  }

  function formatLaneTitle(title) {
    return title.replace("->", "\u2192");
  }

  function getLaneSummary(lane) {
    return LANE_SUMMARIES[lane.slug] || "";
  }

  /* --- Bidirectional --- */
  function renderBidirectional(bidir) {
    var container = document.getElementById("bidir-content");
    if (!container || !bidir || bidir.length === 0) return;

    var povToCrime = [];
    var crimeToPov = [];
    bidir.forEach(function (row) {
      if (row.label.indexOf("poverty_to_") === 0) povToCrime.push(row);
      else crimeToPov.push(row);
    });

    var out = "";

    out += '<h4 style="margin-bottom:0.75rem;font-size:1rem;">Poverty \u2192 Crime</h4>';
    out += '<p style="color:var(--text-muted);font-size:0.875rem;margin-bottom:0.75rem;">Does a rise in county poverty rates lead to higher recorded crime the following year? Treatment: lagged poverty rate (%). Outcome: crime rate per 100,000.</p>';
    out += buildBidirCards(povToCrime);

    out += '<h4 style="margin-top:2rem;margin-bottom:0.75rem;font-size:1rem;">Crime \u2192 Poverty</h4>';
    out += '<p style="color:var(--text-muted);font-size:0.875rem;margin-bottom:0.75rem;">Does a rise in recorded crime lead to higher poverty the following year? Treatment: lagged crime rate (per 100,000). Outcome: poverty rate (%). Coefficients are very small because the units differ by orders of magnitude.</p>';
    out += buildBidirCards(crimeToPov);

    container.innerHTML = out;
  }

  function buildBidirCards(rows) {
    var out = "";
    rows.forEach(function (row) {
      out += '<div class="lane-card" style="margin-bottom:1.25rem;">';

      // Header with title and exploratory badge
      out += '<div class="lane-header">';
      out += '<div class="lane-title">' + esc(formatLaneTitle(row.title)) + '</div>';
      out += '<div class="lane-badges"><span class="badge badge-exploratory">Exploratory</span></div>';
      out += '</div>';

      // Unit explanation
      var unitNote = BIDIR_UNITS[row.label] || "";
      if (unitNote) {
        out += '<p style="color:var(--text-muted);font-size:0.8125rem;margin-bottom:0.75rem;line-height:1.5;">' + esc(unitNote) + '</p>';
      }

      // Estimates table with CIs
      out += '<table class="estimates-table"><thead><tr>';
      out += '<th>Method</th><th>Estimate</th><th>p-value</th><th>95% CI</th>';
      out += '</tr></thead><tbody>';

      out += buildEstimateRow("TWFE", row.fe_coefficient, row.fe_p_value, row.fe_ci_lower, row.fe_ci_upper);
      out += buildEstimateRow("DML", row.dml_theta, row.dml_p_value, row.dml_ci_lower, row.dml_ci_upper);

      out += '</tbody></table>';

      // Diagnostic checks (like policy lanes)
      out += '<div class="checks-row">';

      // Overlap check
      if (row.overlap) {
        var smd = row.overlap.max_abs_smd;
        var ovlpStatus = smd < 0.75 ? "check-pass" : smd < 1.5 ? "check-warn" : "check-fail";
        out += '<span class="check-pill ' + ovlpStatus + '" title="max |SMD| = ' + smd + '"><span class="dot"></span>Overlap: SMD ' + smd + '</span>';
      }

      // Placebo lead check
      if (row.placebo_lead) {
        var plP = row.placebo_lead.p_value;
        var plStatus = plP > 0.05 ? "check-pass" : "check-fail";
        var plLabel = plP > 0.05 ? "Placebo: clean" : "Placebo: significant";
        out += '<span class="check-pill ' + plStatus + '" title="placebo lead p = ' + formatP(plP) + '"><span class="dot"></span>' + plLabel + ' (p=' + formatP(plP) + ')</span>';
      }

      out += '</div>';

      // Summary
      var summary = buildBidirSummary(row);
      if (summary) {
        out += '<div class="lane-summary">' + esc(summary) + '</div>';
      }

      out += '</div>';
    });
    return out;
  }

  function buildBidirSummary(row) {
    var parts = [];
    var feS = row.fe_p_value < 0.05;
    var dmlS = row.dml_p_value < 0.05;

    if (feS && dmlS) parts.push("Both TWFE and DML detect a statistically significant association.");
    else if (dmlS) parts.push("TWFE does not detect a clear association, but DML is statistically significant.");
    else if (feS) parts.push("TWFE detects an association, but DML does not confirm it.");
    else parts.push("Neither TWFE nor DML detects a clear association.");

    if (row.placebo_lead) {
      if (row.placebo_lead.p_value < 0.05) {
        parts.push("The placebo lead is also significant (p=" + formatP(row.placebo_lead.p_value) + "), raising concern that the association reflects shared trends rather than a causal lag.");
      } else {
        parts.push("The placebo lead is clean (p=" + formatP(row.placebo_lead.p_value) + "), supporting the lag structure.");
      }
    }

    if (row.overlap) {
      if (row.overlap.max_abs_smd > 1.5) {
        parts.push("Covariate imbalance is high (max SMD = " + row.overlap.max_abs_smd + "), limiting causal confidence.");
      } else if (row.overlap.max_abs_smd > 0.75) {
        parts.push("Covariate imbalance is moderate (max SMD = " + row.overlap.max_abs_smd + ").");
      }
    }

    return parts.join(" ");
  }

  /* --- Sources --- */
  function renderSources(sources) {
    var container = document.getElementById("sources-table-body");
    if (!container || !sources) return;

    var out = "";
    sources.forEach(function (src) {
      out += "<tr>";
      out += '<td class="source-name">' + esc(src.label) + "</td>";
      out += '<td class="source-desc">' + esc(src.description) + "</td>";
      out += "<td>";
      out +=
        '<span class="coverage-bar-bg"><span class="coverage-bar-fill" style="width:' +
        src.coverage_pct +
        '%"></span></span>';
      out += src.coverage_pct + "%";
      out += "</td>";
      out += "</tr>";
    });
    container.innerHTML = out;
  }

  /* --- Helpers --- */
  function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function formatNum(n) {
    if (n === null || n === undefined) return "\u2014";
    if (Math.abs(n) < 0.001 && n !== 0) return n.toExponential(2);
    if (Math.abs(n) >= 100) return n.toFixed(1);
    if (Math.abs(n) >= 1) return n.toFixed(2);
    return n.toFixed(4);
  }

  function formatP(p) {
    if (p === null || p === undefined) return "\u2014";
    if (p < 0.001) return "< 0.001";
    if (p < 0.01) return p.toFixed(3);
    return p.toFixed(2);
  }

  function esc(s) {
    if (!s) return "";
    var d = document.createElement("div");
    d.appendChild(document.createTextNode(s));
    return d.innerHTML;
  }
})();
