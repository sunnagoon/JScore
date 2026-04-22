const state = {
  snapshot: null,
  filterText: "",
  statFilterText: "",
  selectedTeam: null,
  selectedStatId: null,
  selectedMatchupId: null,
  rrgPayload: null,
  rrgMetric: "power",
  rrgLookback: 120,
  rrgTrail: 7,
  rrgSize: 100,
  rrgYMin: null,
  rrgYMax: null,
  rrgShowTrails: true,
  rrgSelectedTeams: [],
  rrgTeamUniverse: [],
  rrgTeamSelectionInitialized: false,
  rrgTeamMenuOpen: false,
  rrgMomentumLeaders: [],
  rrgMomentumMeta: null,
  starterCompareOpen: {},
};

const summaryCards = document.getElementById("summary-cards");
const teamsBody = document.getElementById("teams-body");
const rankingHead = document.getElementById("ranking-head");
const gamesList = document.getElementById("games-list");
const predictionBars = document.getElementById("prediction-bars");
const valueFinderList = document.getElementById("value-finder-list");
const topTeamsChart = document.getElementById("top-teams-chart");
const refreshBtn = document.getElementById("refresh-btn");
const refreshTime = document.getElementById("refresh-time");
const teamFilter = document.getElementById("team-filter");
const statFilter = document.getElementById("stat-filter");
const detailTeamTitle = document.getElementById("detail-team-title");
const teamHighlights = document.getElementById("team-highlights");
const detailStatsGrid = document.getElementById("detail-stats-grid");
const errorBox = document.getElementById("error-box");
const statSidebar = document.getElementById("stat-ranking-sidebar");
const statSidebarTitle = document.getElementById("stat-sidebar-title");
const statSidebarSubtitle = document.getElementById("stat-sidebar-subtitle");
const statSidebarBody = document.getElementById("stat-sidebar-body");
const statSidebarClose = document.getElementById("stat-sidebar-close");
const matchupSidebar = document.getElementById("matchup-breakdown-sidebar");
const matchupSidebarTitle = document.getElementById("matchup-sidebar-title");
const matchupSidebarSubtitle = document.getElementById("matchup-sidebar-subtitle");
const matchupSidebarBody = document.getElementById("matchup-sidebar-body");
const matchupSidebarClose = document.getElementById("matchup-sidebar-close");
const matchupExportPngBtn = document.getElementById("matchup-export-png");
const diagSummary = document.getElementById("diag-summary");
const diagWinImportance = document.getElementById("diag-win-importance");
const diagWinAblation = document.getElementById("diag-win-ablation");
const diagClvList = document.getElementById("diag-clv-list");
const diagMscoreImportance = document.getElementById("diag-mscore-importance");
const diagMscoreAblation = document.getElementById("diag-mscore-ablation");
const diagCalibrationSummary = document.getElementById("diag-calibration-summary");
const diagReliabilitySvg = document.getElementById("diag-reliability-svg");
const diagBrierDecomp = document.getElementById("diag-brier-decomp");
const rrgMetricSelect = document.getElementById("rrg-metric");
const rrgLookback = document.getElementById("rrg-lookback");
const rrgLookbackLabel = document.getElementById("rrg-lookback-label");
const rrgTrailRange = document.getElementById("rrg-trail-range");
const rrgSizeSelect = document.getElementById("rrg-size");
const rrgYMinInput = document.getElementById("rrg-y-min");
const rrgYMaxInput = document.getElementById("rrg-y-max");
const rrgShowTrails = document.getElementById("rrg-show-trails");
const rrgTeamPicker = document.getElementById("rrg-team-picker");
const rrgTeamToggle = document.getElementById("rrg-team-toggle");
const rrgTeamMenu = document.getElementById("rrg-team-menu");
const rrgTeamOptions = document.getElementById("rrg-team-options");
const rrgTeamAllBtn = document.getElementById("rrg-team-all");
const rrgTeamNoneBtn = document.getElementById("rrg-team-none");
const rrgValidation = document.getElementById("rrg-validation");
const rrgLegend = document.getElementById("rrg-legend");
const rrgSvg = document.getElementById("rrg-svg");
const rrgChartWrap = document.getElementById("rrg-chart-wrap");
const rrgExportPngBtn = document.getElementById("rrg-export-png");
const rrgMomentumNote = document.getElementById("rrg-momentum-note");
const rrgMomentumBoards = document.getElementById("rrg-momentum-boards");
const rrgMomentumExportPngBtn = document.getElementById("rrg-momentum-export-png");
const advancedSummary = document.getElementById("advanced-summary");
const advancedScatterSvg = document.getElementById("advanced-scatter-svg");
const advancedExportPngBtn = document.getElementById("advanced-export-png");
const advancedTeamTitle = document.getElementById("advanced-team-title");
const advancedTeamBars = document.getElementById("advanced-team-bars");
const advancedExpectedGrid = document.getElementById("advanced-expected-grid");

if (rrgMetricSelect) {
  state.rrgMetric = rrgMetricSelect.value;
}
if (rrgLookback) {
  state.rrgLookback = Number(rrgLookback.value);
}
if (rrgTrailRange) {
  state.rrgTrail = Number(rrgTrailRange.value);
}
if (rrgSizeSelect) {
  state.rrgSize = Number(rrgSizeSelect.value);
}
if (rrgYMinInput) {
  const raw = String(rrgYMinInput.value ?? "").trim();
  state.rrgYMin = raw ? Number(raw) : null;
  if (!Number.isFinite(state.rrgYMin)) {
    state.rrgYMin = null;
  }
}
if (rrgYMaxInput) {
  const raw = String(rrgYMaxInput.value ?? "").trim();
  state.rrgYMax = raw ? Number(raw) : null;
  if (!Number.isFinite(state.rrgYMax)) {
    state.rrgYMax = null;
  }
}
if (rrgShowTrails) {
  state.rrgShowTrails = Boolean(rrgShowTrails.checked);
}

function formatNumber(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function toRatioValue(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return null;
  }
  let v = Number(value);
  if (!Number.isFinite(v)) {
    return null;
  }
  if (Math.abs(v) > 1.5) {
    v /= 100;
  }
  if (Math.abs(v) > 1.5) {
    v /= 100;
  }
  return v;
}

function average(values) {
  const nums = (values ?? []).filter((v) => Number.isFinite(Number(v))).map((v) => Number(v));
  if (!nums.length) {
    return null;
  }
  return nums.reduce((sum, v) => sum + v, 0) / nums.length;
}

function percentileRank(values, value, higherBetter = true) {
  const nums = (values ?? []).filter((v) => Number.isFinite(Number(v))).map((v) => Number(v));
  if (!nums.length || !Number.isFinite(Number(value))) {
    return null;
  }
  const v = Number(value);
  const less = nums.filter((n) => n < v).length;
  const equal = nums.filter((n) => n === v).length;
  const rawPct = ((less + (0.5 * equal)) / nums.length) * 100;
  const pct = higherBetter ? rawPct : (100 - rawPct);
  return Math.max(0, Math.min(100, pct));
}

function formatPctPoints(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  const n = Number(value);
  return `${n > 0 ? "+" : ""}${n.toFixed(digits)} pctile`;
}

function signedDelta(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return { text: "-", cls: "delta-flat" };
  }
  if (Number(value) > 0) {
    return { text: `+${value}`, cls: "delta-up" };
  }
  if (Number(value) < 0) {
    return { text: `${value}`, cls: "delta-down" };
  }
  return { text: "0", cls: "delta-flat" };
}

function safeText(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function normalizeText(value) {
  return String(value ?? "").toLowerCase();
}

function numberOrNull(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return null;
  }
  return n;
}

function statId(sourceKind, key) {
  return `${sourceKind}::${key}`;
}

function findTeam(snapshot, teamName) {
  return (snapshot.teams ?? []).find((team) => team.team === teamName);
}

function ensureSelectedTeam(snapshot) {
  const teams = snapshot.teams ?? [];
  if (!teams.length) {
    state.selectedTeam = null;
    return;
  }

  if (!state.selectedTeam || !findTeam(snapshot, state.selectedTeam)) {
    state.selectedTeam = teams[0].team;
  }
}

function formatMetricValue(label, value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    if (value === null || value === undefined) {
      return "-";
    }
    return safeText(value);
  }

  if (typeof value !== "number") {
    return safeText(value);
  }

  const key = normalizeText(label);

  if (key.includes("rank") || key.includes("wins") || key.includes("losses") || key.includes("games played") || key.includes("runs scored") || key.includes("runs allowed") || key.includes("innings pitched") || key === "w" || key === "l") {
    return String(Math.round(value));
  }

  if (key.includes("%") || key.includes("pct") || key.includes("percentage")) {
    if (Math.abs(value) <= 1.2) {
      return `${(value * 100).toFixed(1)}%`;
    }
    return `${value.toFixed(1)}%`;
  }

  if (Math.abs(value) < 1) {
    return value.toFixed(3);
  }

  if (Math.abs(value) >= 100) {
    return value.toFixed(1);
  }

  return value.toFixed(2);
}

function isNumeric(value) {
  return value !== null && value !== undefined && !Number.isNaN(Number(value));
}

function shouldSortAscending(descriptor) {
  const text = `${normalizeText(descriptor.key)} ${normalizeText(descriptor.label)}`;
  return (
    text.includes("rank")
    || text.includes("loss")
    || text.includes("era")
    || text.includes("whip")
    || text.includes("allowed")
    || text.includes("against")
    || text.includes("games back")
  );
}

function getAllStatDescriptors(snapshot) {
  const descriptors = [];

  (snapshot.stat_catalog?.ranking_table ?? []).forEach((entry) => {
    if (entry.key === "team") {
      return;
    }

    descriptors.push({
      id: statId("ranking", entry.key),
      key: entry.key,
      label: entry.label,
      description: entry.description,
      group: "Ranking Table",
      source: "Ranking Table",
      sourceKind: "ranking",
    });
  });

  (snapshot.stat_catalog?.live ?? []).forEach((entry) => {
    descriptors.push({
      id: statId("live", entry.key),
      ...entry,
      source: "Live Model",
      sourceKind: "live",
    });
  });

  (snapshot.stat_catalog?.mlb_api ?? []).forEach((entry) => {
    descriptors.push({
      id: statId("mlb_api", entry.key),
      ...entry,
      source: "MLB Stats API",
      sourceKind: "mlb_api",
    });
  });

  (snapshot.stat_catalog?.workbook ?? []).forEach((entry) => {
    descriptors.push({
      id: statId("workbook", entry.key),
      ...entry,
      source: `Workbook (${entry.sheet ?? "Mscore"})`,
      sourceKind: "workbook",
    });
  });

  return descriptors;
}

function getDescriptorMap(snapshot) {
  return new Map(getAllStatDescriptors(snapshot).map((descriptor) => [descriptor.id, descriptor]));
}

function getStatValue(team, descriptor) {
  if (descriptor.sourceKind === "ranking") {
    if (descriptor.key === "record") {
      return team.wins;
    }
    return team[descriptor.key];
  }

  if (descriptor.sourceKind === "live") {
    return team.live_stats?.[descriptor.key];
  }

  if (descriptor.sourceKind === "mlb_api") {
    return team.mlb_api_stats?.[descriptor.key];
  }

  if (descriptor.sourceKind === "workbook") {
    return team.workbook_stats?.[descriptor.key];
  }

  return null;
}

function openStatSidebar(snapshot, descriptorId) {
  if (!snapshot) {
    return;
  }
  const descriptorMap = getDescriptorMap(snapshot);
  if (!descriptorMap.has(descriptorId)) {
    return;
  }

  state.selectedStatId = descriptorId;
  renderStatRankingSidebar(snapshot);
}

function closeStatSidebar() {
  state.selectedStatId = null;
  statSidebar.classList.add("is-closed");
  statSidebar.setAttribute("aria-hidden", "true");
}

function renderStatRankingSidebar(snapshot) {
  if (!state.selectedStatId || !snapshot) {
    statSidebar.classList.add("is-closed");
    statSidebar.setAttribute("aria-hidden", "true");
    return;
  }

  const descriptor = getDescriptorMap(snapshot).get(state.selectedStatId);
  if (!descriptor) {
    closeStatSidebar();
    return;
  }

  const rows = (snapshot.teams ?? [])
    .map((team) => ({
      team: team.team,
      value: getStatValue(team, descriptor),
      selected: team.team === state.selectedTeam,
    }))
    .filter((item) => item.value !== null && item.value !== undefined && item.value !== "-");

  const hasNumeric = rows.some((item) => isNumeric(item.value));
  const ascending = shouldSortAscending(descriptor);

  rows.sort((a, b) => {
    if (hasNumeric && isNumeric(a.value) && isNumeric(b.value)) {
      return ascending ? Number(a.value) - Number(b.value) : Number(b.value) - Number(a.value);
    }

    const left = String(a.value ?? "");
    const right = String(b.value ?? "");
    return left.localeCompare(right);
  });

  statSidebarTitle.textContent = `${descriptor.label} Team Rankings`;
  statSidebarSubtitle.textContent = descriptor.description || "Team ranking for selected statistic.";

  if (!rows.length) {
    statSidebarBody.innerHTML = "<p class='muted-note'>No values available for this statistic.</p>";
  } else {
    statSidebarBody.innerHTML = rows
      .map(
        (row, index) => `
        <article class="stat-rank-row ${row.selected ? "selected" : ""}">
          <span class="stat-rank-pos">#${index + 1}</span>
          <span class="stat-rank-team">${safeText(row.team)}</span>
          <span class="stat-rank-value">${safeText(formatMetricValue(descriptor.label, row.value))}</span>
        </article>
      `,
      )
      .join("");
  }

  statSidebar.classList.remove("is-closed");
  statSidebar.setAttribute("aria-hidden", "false");
}



function getMatchupId(game) {
  if (game && game.game_pk !== null && game.game_pk !== undefined) {
    return `pk:${game.game_pk}`;
  }
  return `alt:${game?.official_date ?? ""}|${game?.away_team ?? ""}|${game?.home_team ?? ""}`;
}

function findMatchupById(snapshot, matchupId) {
  if (!snapshot || !matchupId) {
    return null;
  }
  const games = snapshot.matchup_predictions ?? [];
  return games.find((game) => getMatchupId(game) === matchupId) ?? null;
}

function formatSignedPctPoints(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  const num = Number(value) * 100;
  if (num > 0) {
    return `+${num.toFixed(1)} pts`;
  }
  return `${num.toFixed(1)} pts`;
}

function getTeamMlbValue(team, key) {
  return team?.mlb_api_stats?.[key];
}

function getTeamLiveValue(team, key) {
  const fromLive = team?.live_stats?.[key];
  if (fromLive !== null && fromLive !== undefined) {
    return fromLive;
  }
  return team?.[key];
}

function starterMetricWinner(awayValue, homeValue, lowerBetter = false) {
  if (!isNumeric(awayValue) || !isNumeric(homeValue)) {
    return "none";
  }

  const awayNum = Number(awayValue);
  const homeNum = Number(homeValue);
  const epsilon = 1e-9;
  if (Math.abs(awayNum - homeNum) <= epsilon) {
    return "tie";
  }

  if (lowerBetter) {
    return awayNum < homeNum ? "away" : "home";
  }
  return awayNum > homeNum ? "away" : "home";
}

function formatStarterMetric(row, value) {
  if (!isNumeric(value)) {
    return "-";
  }

  const num = Number(value);
  if (row.format === "percent") {
    return formatPercent(num, row.digits ?? 1);
  }
  return formatNumber(num, row.digits ?? 2);
}

function buildStarterComparisonRows(game) {
  return [
    { label: "ERA", awayValue: game?.away_starter_era, homeValue: game?.home_starter_era, lowerBetter: true, digits: 2 },
    { label: "xFIP", awayValue: game?.away_starter_xfip, homeValue: game?.home_starter_xfip, lowerBetter: true, digits: 2 },
    { label: "WHIP", awayValue: game?.away_starter_whip, homeValue: game?.home_starter_whip, lowerBetter: true, digits: 3 },
    { label: "K-BB / 9", awayValue: game?.away_starter_k_minus_bb_per9, homeValue: game?.home_starter_k_minus_bb_per9, lowerBetter: false, digits: 2 },
    { label: "QS%", awayValue: game?.away_starter_qs_rate, homeValue: game?.home_starter_qs_rate, lowerBetter: false, format: "percent", digits: 1 },
    { label: "IP", awayValue: game?.away_starter_innings_pitched, homeValue: game?.home_starter_innings_pitched, lowerBetter: false, digits: 1 },
    { label: "Reliability", awayValue: game?.away_starter_reliability, homeValue: game?.home_starter_reliability, lowerBetter: false, format: "percent", digits: 1 },
  ];
}

function renderStarterComparisonPanel(game, matchupId, isOpen) {
  const awayName = game?.away_probable_pitcher ?? game?.away_starter_name ?? "TBD";
  const homeName = game?.home_probable_pitcher ?? game?.home_starter_name ?? "TBD";
  const awayHand = game?.away_starter_hand ? ` (${game.away_starter_hand})` : "";
  const homeHand = game?.home_starter_hand ? ` (${game.home_starter_hand})` : "";
  const rows = buildStarterComparisonRows(game);

  const hasAnyMetric = rows.some((row) => isNumeric(row.awayValue) || isNumeric(row.homeValue));

  const body = !hasAnyMetric
    ? "<p class='muted-note'>Starter metrics unavailable until probable pitcher data is published.</p>"
    : rows
      .map((row) => {
        const winner = starterMetricWinner(row.awayValue, row.homeValue, row.lowerBetter);
        const awayClass = winner === "away" ? "pitcher-metric-win" : (winner === "tie" ? "pitcher-metric-tie" : "");
        const homeClass = winner === "home" ? "pitcher-metric-win" : (winner === "tie" ? "pitcher-metric-tie" : "");

        return `
          <div class="sp-compare-row">
            <span class="sp-compare-value away ${awayClass}">${safeText(formatStarterMetric(row, row.awayValue))}</span>
            <span class="sp-compare-label">${safeText(row.label)}</span>
            <span class="sp-compare-value home ${homeClass}">${safeText(formatStarterMetric(row, row.homeValue))}</span>
          </div>
        `;
      })
      .join("");

  return `
    <div class="sp-compare-panel ${isOpen ? "" : "is-closed"}" data-sp-panel="${safeText(matchupId)}">
      <div class="sp-compare-head">
        <span class="sp-compare-name away">${safeText(awayName)}${safeText(awayHand)}</span>
        <span class="sp-compare-center">SP Compare</span>
        <span class="sp-compare-name home">${safeText(homeName)}${safeText(homeHand)}</span>
      </div>
      <div class="sp-compare-grid">${body}</div>
    </div>
  `;
}

function toggleStarterComparison(matchupId) {
  if (!matchupId) {
    return;
  }

  state.starterCompareOpen = state.starterCompareOpen ?? {};
  state.starterCompareOpen[matchupId] = !Boolean(state.starterCompareOpen[matchupId]);

  if (state.snapshot) {
    renderPredictions(state.snapshot);
  }
}

function buildMatchupComponentRows(game) {
  return [
    { label: "Starter", value: Number(game?.starter_adjustment ?? 0) },
    { label: "Bullpen Core", value: Number(game?.bullpen_adjustment ?? 0) },
    { label: "Bullpen Health", value: Number(game?.bullpen_health_adjustment ?? 0) },
    { label: "Split", value: Number(game?.split_adjustment ?? 0) },
    { label: "Travel", value: Number(game?.travel_adjustment ?? 0) },
    { label: "Lineup Core", value: Number(game?.lineup_adjustment ?? 0) },
    { label: "Lineup Health", value: Number(game?.lineup_health_adjustment ?? 0) },
    { label: "Luck/Regression", value: Number(game?.luck_adjustment ?? 0) },
    { label: "Expected Quality", value: Number(game?.advanced_adjustment ?? 0) },
  ];
}

function clampProb(value, lo = 0.02, hi = 0.98) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return 0.5;
  }
  return Math.max(lo, Math.min(hi, n));
}

function getPreMarketHomeProb(game) {
  const explicit = Number(game?.pre_market_home_win_prob);
  if (Number.isFinite(explicit)) {
    return clampProb(explicit);
  }

  const modelHome = Number(game?.model_home_win_prob ?? game?.home_win_prob ?? 0.5);
  const contextAdj = buildMatchupComponentRows(game).reduce((sum, row) => sum + Number(row.value ?? 0), 0);
  return clampProb(modelHome + contextAdj);
}

function getModelPickTeam(game) {
  if (game?.model_pick_team) {
    return game.model_pick_team;
  }

  const homeProb = Number(game?.home_win_prob ?? 0.5);
  return homeProb >= 0.5 ? game?.home_team : game?.away_team;
}

function getPickProbability(homeProb, pickTeam, homeTeam, awayTeam) {
  if (pickTeam === homeTeam) {
    return Number(homeProb);
  }
  if (pickTeam === awayTeam) {
    return 1 - Number(homeProb);
  }
  return Math.max(Number(homeProb), 1 - Number(homeProb));
}

function computeValueTier(edge) {
  if (edge === null || edge === undefined || Number.isNaN(Number(edge))) {
    return "none";
  }

  const magnitude = Math.abs(Number(edge));
  if (magnitude >= 0.08) return "elite";
  if (magnitude >= 0.05) return "strong";
  if (magnitude >= 0.03) return "actionable";
  if (magnitude >= 0.015) return "thin";
  return "none";
}

function formatValueTier(tier) {
  const key = String(tier || "none").toLowerCase();
  if (key === "elite") return "Elite";
  if (key === "strong") return "Strong";
  if (key === "actionable") return "Action";
  if (key === "thin") return "Thin";
  return "No Edge";
}

function formatBetQualityGrade(grade) {
  const key = String(grade || "pass").trim().toLowerCase();
  if (key === "a") return "A";
  if (key === "b") return "B";
  if (key === "c") return "C";
  return "Pass";
}

function computeBetQualityFallback({
  preMarketHome,
  marketEdgePick,
  marketEdgePickRaw,
  confidence,
  uncertaintyLevel,
  uncertaintyMultiplier,
  bandHalf,
  marketAvailable,
}) {
  const pre = clampProb(Number(preMarketHome ?? 0.5));
  const edgeRaw = Number(marketEdgePickRaw);
  const edge = Number(marketEdgePick);

  let edgeBasis = null;
  if (Number.isFinite(edgeRaw)) {
    edgeBasis = Math.abs(edgeRaw);
  } else if (Number.isFinite(edge)) {
    edgeBasis = Math.abs(edge);
  } else {
    edgeBasis = Math.abs(pre - 0.5) * 0.85;
  }

  const edgeScale = marketAvailable ? 0.08 : 0.05;
  const edgeStrength = Math.max(0, Math.min(1, Number(edgeBasis) / edgeScale));
  const confClamped = Math.max(0, Math.min(1, Number(confidence ?? 0)));
  const confidenceFactor = 0.55 + (0.45 * confClamped);

  const band = Number(bandHalf);
  let bandFactor = 0.9;
  if (Number.isFinite(band)) {
    const bandRef = Math.max(0, Math.min(0.2, band));
    bandFactor = 1 - ((bandRef / 0.2) * 0.55);
  }
  bandFactor = Math.max(0.35, Math.min(1, bandFactor));

  const level = String(uncertaintyLevel || "low").trim().toLowerCase();
  const levelFactor = level === "high" ? 0.58 : (level === "medium" ? 0.8 : 1.0);

  const marketFactor = marketAvailable ? 1.0 : 0.9;
  const uncMultRaw = Number(uncertaintyMultiplier);
  const uncMult = Number.isFinite(uncMultRaw) ? Math.max(0.35, Math.min(1, uncMultRaw)) : 1.0;

  let score = 100 * edgeStrength * confidenceFactor * bandFactor * levelFactor * marketFactor * uncMult;
  score = Math.max(0, Math.min(99.9, score));

  const edgeForGate = Number.isFinite(edge) ? Math.abs(edge) : Math.abs(pre - 0.5);
  const actionableCut = marketAvailable ? 60 : 42;
  const actionable = score >= actionableCut && level !== "high" && edgeForGate >= 0.02;

  return {
    score: Number(score.toFixed(1)),
    grade: formatBetQualityGrade(score >= 75 ? "A" : (score >= 60 ? "B" : (score >= 45 ? "C" : "Pass"))),
    actionable,
  };
}

function sortValueFinderRows(a, b) {
  const qualityA = Number(a.bet_quality_score ?? NaN);
  const qualityB = Number(b.bet_quality_score ?? NaN);
  if (Number.isFinite(qualityA) || Number.isFinite(qualityB)) {
    const qa = Number.isFinite(qualityA) ? qualityA : -1;
    const qb = Number.isFinite(qualityB) ? qualityB : -1;
    if (qa !== qb) return qb - qa;
  }

  const edgeA = a.market_edge_pick === null || a.market_edge_pick === undefined ? -1 : Math.abs(Number(a.market_edge_pick));
  const edgeB = b.market_edge_pick === null || b.market_edge_pick === undefined ? -1 : Math.abs(Number(b.market_edge_pick));
  if (edgeA !== edgeB) return edgeB - edgeA;

  return Number(b.confidence || 0) - Number(a.confidence || 0);
}

function buildValueFinderRows(snapshot) {
  const board = snapshot.prediction_value_board ?? [];
  const matchupGames = snapshot.matchup_predictions ?? [];
  const gameById = new Map(matchupGames.map((game) => [getMatchupId(game), game]));

  if (board.length) {
    return board
      .map((row) => {
        const matchupId = row.matchup_id ?? getMatchupId(row);
        const linkedGame = gameById.get(matchupId);
        const modelPickTeam = row.model_pick_team ?? (linkedGame ? getModelPickTeam(linkedGame) : null);

        let finalPickProb = null;
        if (linkedGame) {
          const finalHomeProb = Number(linkedGame.home_win_prob ?? NaN);
          if (Number.isFinite(finalHomeProb) && modelPickTeam) {
            finalPickProb = getPickProbability(finalHomeProb, modelPickTeam, linkedGame.home_team, linkedGame.away_team);
          }
        }

        const marketPickProb = row.market_pick_prob === null || row.market_pick_prob === undefined
          ? null
          : Number(row.market_pick_prob);

        const confidence = Number(row.confidence ?? NaN);
        const preMarketHome = linkedGame ? getPreMarketHomeProb(linkedGame) : 0.5;
        const fallbackQuality = computeBetQualityFallback({
          preMarketHome,
          marketEdgePick: row.market_edge_pick,
          marketEdgePickRaw: row.market_edge_pick_raw,
          confidence: Number.isFinite(confidence) ? confidence : 0,
          uncertaintyLevel: String(row.uncertainty_level ?? "low"),
          uncertaintyMultiplier: Number(row.uncertainty_edge_multiplier ?? 1),
          bandHalf: row.home_win_prob_band_half,
          marketAvailable: marketPickProb !== null,
        });

        const rawQualityScore = Number(row.bet_quality_score ?? NaN);
        const betQualityScore = Number.isFinite(rawQualityScore) ? rawQualityScore : fallbackQuality.score;
        const betQualityGrade = row.bet_quality_grade ? formatBetQualityGrade(row.bet_quality_grade) : fallbackQuality.grade;
        const betQualityActionable = row.bet_quality_actionable === undefined || row.bet_quality_actionable === null
          ? Boolean(fallbackQuality.actionable)
          : Boolean(row.bet_quality_actionable);

        const linkedDelta = linkedGame ?? {};
        const pick = (key) => (row[key] === undefined || row[key] === null ? linkedDelta[key] : row[key]);

        return {
          matchup_id: matchupId,
          away_team: row.away_team,
          home_team: row.home_team,
          model_pick_team: modelPickTeam,
          model_pick_prob: Number(row.model_pick_prob ?? NaN),
          final_pick_prob: finalPickProb,
          market_pick_prob: marketPickProb,
          market_edge_pick: row.market_edge_pick === null || row.market_edge_pick === undefined ? null : Number(row.market_edge_pick),
          market_edge_pick_raw: row.market_edge_pick_raw === null || row.market_edge_pick_raw === undefined ? null : Number(row.market_edge_pick_raw),
          uncertainty_edge_multiplier: Number(row.uncertainty_edge_multiplier ?? 1),
          uncertainty_level: String(row.uncertainty_level ?? "low"),
          high_uncertainty: Boolean(row.high_uncertainty),
          home_win_prob_band_low: row.home_win_prob_band_low === null || row.home_win_prob_band_low === undefined ? null : Number(row.home_win_prob_band_low),
          home_win_prob_band_high: row.home_win_prob_band_high === null || row.home_win_prob_band_high === undefined ? null : Number(row.home_win_prob_band_high),
          home_win_prob_band_half: row.home_win_prob_band_half === null || row.home_win_prob_band_half === undefined ? null : Number(row.home_win_prob_band_half),
          uncertainty_note: row.uncertainty_note ?? null,
          confidence,
          value_tier: row.value_tier ?? computeValueTier(row.market_edge_pick),
          is_market_upset: Boolean(row.is_market_upset),
          market_available: marketPickProb !== null,
          bet_quality_score: betQualityScore,
          bet_quality_grade: betQualityGrade,
          bet_quality_actionable: betQualityActionable,
          delta_source_date: pick("delta_source_date") ?? null,
          delta_days: numberOrNull(pick("delta_days")),
          delta_home_win_prob: numberOrNull(pick("delta_home_win_prob")),
          delta_pre_market_home_win_prob: numberOrNull(pick("delta_pre_market_home_win_prob")),
          delta_model_home_win_prob: numberOrNull(pick("delta_model_home_win_prob")),
          delta_market_home_win_prob: numberOrNull(pick("delta_market_home_win_prob")),
          delta_favored_team_changed: pick("delta_favored_team_changed") === null || pick("delta_favored_team_changed") === undefined ? null : Boolean(pick("delta_favored_team_changed")),
        };
      })
      .sort(sortValueFinderRows);
  }

  return matchupGames
    .map((game) => {
      const homeTeam = game.home_team;
      const awayTeam = game.away_team;
      const homeProb = Number(game.home_win_prob ?? NaN);
      if (!Number.isFinite(homeProb)) {
        return null;
      }

      const preMarketHome = getPreMarketHomeProb(game);
      const modelPickTeam = getModelPickTeam(game);
      const modelPickProb = getPickProbability(preMarketHome, modelPickTeam, homeTeam, awayTeam);
      const finalPickProb = getPickProbability(homeProb, modelPickTeam, homeTeam, awayTeam);

      const marketHome = game.market_home_win_prob === null || game.market_home_win_prob === undefined
        ? null
        : Number(game.market_home_win_prob);

      const marketPickProb = marketHome === null
        ? null
        : getPickProbability(marketHome, modelPickTeam, homeTeam, awayTeam);

      const marketEdgePick = game.model_vs_market_edge_pick === null || game.model_vs_market_edge_pick === undefined
        ? (marketPickProb === null ? null : modelPickProb - marketPickProb)
        : Number(game.model_vs_market_edge_pick);

      const marketFavTeam = marketHome === null
        ? null
        : (game.market_favored_team ?? (marketHome >= 0.5 ? homeTeam : awayTeam));

      const confidence = Math.abs(homeProb - 0.5) * 2;
      const fallbackQuality = computeBetQualityFallback({
        preMarketHome,
        marketEdgePick: marketEdgePick,
        marketEdgePickRaw: game.model_vs_market_edge_pick === null || game.model_vs_market_edge_pick === undefined ? marketEdgePick : Number(game.model_vs_market_edge_pick),
        confidence,
        uncertaintyLevel: String(game.uncertainty_level ?? "low"),
        uncertaintyMultiplier: Number(game.uncertainty_edge_multiplier ?? 1),
        bandHalf: game.home_win_prob_band_half,
        marketAvailable: marketPickProb !== null,
      });

      return {
        matchup_id: getMatchupId(game),
        away_team: awayTeam,
        home_team: homeTeam,
        model_pick_team: modelPickTeam,
        model_pick_prob: modelPickProb,
        final_pick_prob: finalPickProb,
        market_pick_prob: marketPickProb,
        market_edge_pick: marketEdgePick,
        market_edge_pick_raw: game.model_vs_market_edge_pick === null || game.model_vs_market_edge_pick === undefined ? marketEdgePick : Number(game.model_vs_market_edge_pick),
        uncertainty_edge_multiplier: Number(game.uncertainty_edge_multiplier ?? 1),
        uncertainty_level: String(game.uncertainty_level ?? "low"),
        high_uncertainty: Boolean(game.high_uncertainty),
        home_win_prob_band_low: game.home_win_prob_band_low === null || game.home_win_prob_band_low === undefined ? null : Number(game.home_win_prob_band_low),
        home_win_prob_band_high: game.home_win_prob_band_high === null || game.home_win_prob_band_high === undefined ? null : Number(game.home_win_prob_band_high),
        home_win_prob_band_half: game.home_win_prob_band_half === null || game.home_win_prob_band_half === undefined ? null : Number(game.home_win_prob_band_half),
        uncertainty_note: game.uncertainty_note ?? null,
        confidence,
        value_tier: game.value_tier ?? computeValueTier(marketEdgePick),
        is_market_upset: Boolean(marketFavTeam && marketFavTeam !== modelPickTeam),
        market_available: marketPickProb !== null,
        bet_quality_score: fallbackQuality.score,
        bet_quality_grade: fallbackQuality.grade,
        bet_quality_actionable: fallbackQuality.actionable,
        delta_source_date: game.delta_source_date ?? null,
        delta_days: numberOrNull(game.delta_days),
        delta_home_win_prob: numberOrNull(game.delta_home_win_prob),
        delta_pre_market_home_win_prob: numberOrNull(game.delta_pre_market_home_win_prob),
        delta_model_home_win_prob: numberOrNull(game.delta_model_home_win_prob),
        delta_market_home_win_prob: numberOrNull(game.delta_market_home_win_prob),
        delta_favored_team_changed: game.delta_favored_team_changed === null || game.delta_favored_team_changed === undefined ? null : Boolean(game.delta_favored_team_changed),
      };
    })
    .filter(Boolean)
    .sort(sortValueFinderRows);
}

function buildWhyPickRows(game) {
  const homeTeam = game?.home_team;
  const awayTeam = game?.away_team;
  const favoredTeam = game?.favored_team ?? homeTeam;
  const favorsHome = favoredTeam === homeTeam;
  const orient = (value) => (favorsHome ? Number(value) : -Number(value));

  const modelHome = Number(game?.model_home_win_prob ?? game?.home_win_prob ?? 0.5);
  const rawCandidate = Number(game?.model_home_win_prob_raw);
  const modelHomeRaw = Number.isFinite(rawCandidate) ? rawCandidate : null;
  const explicitHomePriorAdj = Number(game?.home_prior_neutralization);
  const homePriorAdj = Number.isFinite(explicitHomePriorAdj)
    ? explicitHomePriorAdj
    : (modelHomeRaw === null ? 0 : (modelHome - modelHomeRaw));
  const preMarketHome = getPreMarketHomeProb(game);
  const finalHome = clampProb(Number(game?.home_win_prob ?? preMarketHome));

  const baseRows = modelHomeRaw === null
    ? [{ label: "Base Model", value: orient(modelHome - 0.5) }]
    : [
        { label: "Base Model (Raw)", value: orient(modelHomeRaw - 0.5) },
        { label: "Home Prior Neutralization", value: orient(homePriorAdj) },
      ];

  const rows = [
    ...baseRows,
    ...buildMatchupComponentRows(game).map((row) => ({ label: row.label, value: orient(row.value) })),
  ];

  const marketHome = game?.market_home_win_prob === null || game?.market_home_win_prob === undefined
    ? null
    : Number(game.market_home_win_prob);

  if (marketHome !== null && Number.isFinite(marketHome)) {
    const favoredPre = favorsHome ? preMarketHome : 1 - preMarketHome;
    const favoredFinal = favorsHome ? finalHome : 1 - finalHome;
    const favoredMarket = favorsHome ? marketHome : 1 - marketHome;
    rows.push({ label: "Market Blend", value: favoredFinal - favoredPre });
    rows.push({ label: "Model vs Market", value: favoredPre - favoredMarket });
  }

  return rows
    .map((row) => ({ ...row, magnitude: Math.abs(Number(row.value ?? 0)) }))
    .sort((a, b) => b.magnitude - a.magnitude)
    .slice(0, 5);
}

function buildMatchupDeltaRows(game) {
  const keys = [
    { label: "Final Home", key: "delta_home_win_prob" },
    { label: "Pre-Market Home", key: "delta_pre_market_home_win_prob" },
    { label: "Model Home", key: "delta_model_home_win_prob" },
    { label: "Market Home", key: "delta_market_home_win_prob" },
    { label: "Starter", key: "delta_starter_adjustment" },
    { label: "Split", key: "delta_split_adjustment" },
    { label: "Bullpen Core", key: "delta_bullpen_adjustment" },
    { label: "Bullpen Health", key: "delta_bullpen_health_adjustment" },
    { label: "Lineup Core", key: "delta_lineup_adjustment" },
    { label: "Lineup Health", key: "delta_lineup_health_adjustment" },
    { label: "Travel", key: "delta_travel_adjustment" },
    { label: "Luck/Regression", key: "delta_luck_adjustment" },
    { label: "Expected Quality", key: "delta_advanced_adjustment" },
  ];

  const rows = keys
    .map((entry) => {
      const raw = numberOrNull(game?.[entry.key]);
      if (raw === null) {
        return null;
      }
      return {
        label: entry.label,
        value: raw,
        magnitude: Math.abs(raw),
      };
    })
    .filter(Boolean)
    .sort((a, b) => Number(b.magnitude) - Number(a.magnitude))
    .slice(0, 8);

  const daysRaw = numberOrNull(game?.delta_days);
  return {
    rows,
    sourceDate: game?.delta_source_date ?? null,
    days: (daysRaw !== null && daysRaw > 0) ? daysRaw : null,
    favoredChanged: game?.delta_favored_team_changed === null || game?.delta_favored_team_changed === undefined
      ? null
      : Boolean(game.delta_favored_team_changed),
  };
}

function buildAdvancedWaterfallRows(game) {
  const parts = [
    { label: "Expected Offense", value: Number(game?.advanced_offense_edge ?? 0) },
    { label: "Expected Pitching", value: Number(game?.advanced_pitching_edge ?? 0) },
    { label: "xwOBAcon Edge", value: Number(game?.advanced_xwobacon_edge ?? 0) },
    { label: "xFIP Edge", value: Number(game?.advanced_xfip_edge ?? 0) },
    { label: "Leverage Quality", value: Number(game?.advanced_leverage_edge ?? 0) },
    { label: "Clutch Execution", value: Number(game?.advanced_clutch_edge ?? 0) },
  ];

  let cumulative = 0;
  const rows = parts.map((part) => {
    const start = cumulative;
    cumulative += Number(part.value ?? 0);
    return {
      label: part.label,
      value: Number(part.value ?? 0),
      start,
      end: cumulative,
      cumulative,
    };
  });

  rows.push({
    label: "Advanced Total",
    value: Number(game?.advanced_adjustment ?? 0),
    start: 0,
    end: Number(game?.advanced_adjustment ?? 0),
    cumulative: Number(game?.advanced_adjustment ?? 0),
    total: true,
  });
  return rows;
}


function makeMatchupStatRows(homeTeam, awayTeam, game) {
  return [
    { label: "Record", home: `${homeTeam?.wins ?? "-"}-${homeTeam?.losses ?? "-"}`, away: `${awayTeam?.wins ?? "-"}-${awayTeam?.losses ?? "-"}` },
    { label: "Win %", home: formatPercent(homeTeam?.win_pct, 1), away: formatPercent(awayTeam?.win_pct, 1) },
    { label: "Run Diff", home: formatNumber(homeTeam?.run_diff, 0), away: formatNumber(awayTeam?.run_diff, 0) },
    { label: "Live Power", home: formatNumber(homeTeam?.live_power_score, 2), away: formatNumber(awayTeam?.live_power_score, 2) },
    { label: "D+ Score", home: formatNumber(homeTeam?.d_plus_score, 2), away: formatNumber(awayTeam?.d_plus_score, 2) },
    { label: "Mscore", home: formatNumber(homeTeam?.mscore, 3), away: formatNumber(awayTeam?.mscore, 3) },
    { label: "Projected Wins", home: formatNumber(homeTeam?.projected_wins_162, 1), away: formatNumber(awayTeam?.projected_wins_162, 1) },
    { label: "OPS", home: formatNumber(getTeamMlbValue(homeTeam, "mlb::season::hitting::ops"), 3), away: formatNumber(getTeamMlbValue(awayTeam, "mlb::season::hitting::ops"), 3) },
    { label: "OBP", home: formatNumber(getTeamMlbValue(homeTeam, "mlb::season::hitting::obp"), 3), away: formatNumber(getTeamMlbValue(awayTeam, "mlb::season::hitting::obp"), 3) },
    { label: "SLG", home: formatNumber(getTeamMlbValue(homeTeam, "mlb::season::hitting::slg"), 3), away: formatNumber(getTeamMlbValue(awayTeam, "mlb::season::hitting::slg"), 3) },
    { label: "xwOBAcon Proxy", home: formatNumber(getTeamLiveValue(homeTeam, "xwobacon_proxy") ?? game?.home_xwobacon_proxy, 3), away: formatNumber(getTeamLiveValue(awayTeam, "xwobacon_proxy") ?? game?.away_xwobacon_proxy, 3) },
    { label: "xFIP Proxy", home: formatNumber(getTeamLiveValue(homeTeam, "xfip_proxy") ?? game?.home_xfip_proxy, 3), away: formatNumber(getTeamLiveValue(awayTeam, "xfip_proxy") ?? game?.away_xfip_proxy, 3) },
    { label: "OPS (RISP)", home: formatNumber(getTeamLiveValue(homeTeam, "hitting_ops_risp"), 3), away: formatNumber(getTeamLiveValue(awayTeam, "hitting_ops_risp"), 3) },
    { label: "OPS (RISP,2 Out)", home: formatNumber(getTeamLiveValue(homeTeam, "hitting_ops_risp2"), 3), away: formatNumber(getTeamLiveValue(awayTeam, "hitting_ops_risp2"), 3) },
    { label: "OPS (Late/Close)", home: formatNumber(getTeamLiveValue(homeTeam, "hitting_ops_late_close"), 3), away: formatNumber(getTeamLiveValue(awayTeam, "hitting_ops_late_close"), 3) },
    { label: "Opp OPS (Late/Close)", home: formatNumber(getTeamLiveValue(homeTeam, "pitching_ops_late_close_allowed"), 3), away: formatNumber(getTeamLiveValue(awayTeam, "pitching_ops_late_close_allowed"), 3) },
    { label: "Leverage Net", home: formatNumber(getTeamLiveValue(homeTeam, "leverage_net_quality") ?? game?.home_leverage_net_quality, 1), away: formatNumber(getTeamLiveValue(awayTeam, "leverage_net_quality") ?? game?.away_leverage_net_quality, 1) },
    { label: "Clutch Index", home: formatNumber(getTeamLiveValue(homeTeam, "clutch_index") ?? game?.home_clutch_index, 1), away: formatNumber(getTeamLiveValue(awayTeam, "clutch_index") ?? game?.away_clutch_index, 1) },
    { label: "Joe Score", home: formatNumber(getTeamLiveValue(homeTeam, "joe_score"), 2), away: formatNumber(getTeamLiveValue(awayTeam, "joe_score"), 2) },
    { label: "Joe Band", home: `${formatNumber(getTeamLiveValue(homeTeam, "joe_band_low"), 1)}-${formatNumber(getTeamLiveValue(homeTeam, "joe_band_high"), 1)}`, away: `${formatNumber(getTeamLiveValue(awayTeam, "joe_band_low"), 1)}-${formatNumber(getTeamLiveValue(awayTeam, "joe_band_high"), 1)}` },
    { label: "Joe Confidence", home: isNumeric(getTeamLiveValue(homeTeam, "joe_confidence")) ? `${formatNumber(getTeamLiveValue(homeTeam, "joe_confidence"), 1)}%` : "-", away: isNumeric(getTeamLiveValue(awayTeam, "joe_confidence")) ? `${formatNumber(getTeamLiveValue(awayTeam, "joe_confidence"), 1)}%` : "-" },
    { label: "Joe Uncertainty", home: getTeamLiveValue(homeTeam, "joe_uncertainty_level") ?? "-", away: getTeamLiveValue(awayTeam, "joe_uncertainty_level") ?? "-" },
    { label: "Exp Offense Q", home: formatNumber(game?.home_expected_offense_quality, 1), away: formatNumber(game?.away_expected_offense_quality, 1) },
    { label: "Exp Pitching Q", home: formatNumber(game?.home_expected_pitching_quality, 1), away: formatNumber(game?.away_expected_pitching_quality, 1) },
    { label: "Team ERA", home: formatNumber(getTeamMlbValue(homeTeam, "mlb::season::pitching::era"), 2), away: formatNumber(getTeamMlbValue(awayTeam, "mlb::season::pitching::era"), 2) },
    { label: "Team WHIP", home: formatNumber(getTeamMlbValue(homeTeam, "mlb::season::pitching::whip"), 3), away: formatNumber(getTeamMlbValue(awayTeam, "mlb::season::pitching::whip"), 3) },
    { label: "K / 9", home: formatNumber(getTeamMlbValue(homeTeam, "mlb::season::pitching::strikeoutsPer9Inn"), 2), away: formatNumber(getTeamMlbValue(awayTeam, "mlb::season::pitching::strikeoutsPer9Inn"), 2) },
    { label: "Bullpen Health", home: formatNumber(game?.bullpen_health_home_score, 1), away: formatNumber(game?.bullpen_health_away_score, 1) },
    { label: "Lineup Health", home: formatNumber(game?.lineup_health_home_score, 1), away: formatNumber(game?.lineup_health_away_score, 1) },
    { label: "Luck Index", home: formatNumber(game?.home_luck_index, 4), away: formatNumber(game?.away_luck_index, 4) },
  ];
}

function closeMatchupSidebar() {
  state.selectedMatchupId = null;
  matchupSidebar.classList.add("is-closed");
  matchupSidebar.setAttribute("aria-hidden", "true");
  if (matchupExportPngBtn) {
    matchupExportPngBtn.disabled = true;
  }
}

function openMatchupSidebar(snapshot, matchupId) {
  if (!snapshot || !matchupId) {
    return;
  }
  closeStatSidebar();
  state.selectedMatchupId = matchupId;
  renderMatchupSidebar(snapshot);
  if (state.rrgPayload) {
    renderRrg(state.rrgPayload);
  }
}

function renderMatchupSidebar(snapshot) {
  if (!state.selectedMatchupId || !snapshot) {
    matchupSidebar.classList.add("is-closed");
    matchupSidebar.setAttribute("aria-hidden", "true");
    return;
  }

  const game = findMatchupById(snapshot, state.selectedMatchupId);
  if (!game) {
    closeMatchupSidebar();
    return;
  }

  const homeTeam = findTeam(snapshot, game.home_team);
  const awayTeam = findTeam(snapshot, game.away_team);
  const favoredTeam = game.favored_team ?? game.home_team;
  const favoredProb = Number(game.favored_win_prob ?? 0.5);

  const componentRows = buildMatchupComponentRows(game);

  const contextAdj = componentRows.reduce((sum, row) => sum + row.value, 0);
  const modelHome = Number(game.model_home_win_prob ?? game.home_win_prob ?? 0.5);
  const preMarketHome = getPreMarketHomeProb(game);
  const finalHome = clampProb(Number(game.home_win_prob ?? preMarketHome));
  const marketImpact = finalHome - preMarketHome;

  const deltaRowsData = buildMatchupDeltaRows(game);
  const deltaSummary = numberOrNull(game?.delta_home_win_prob);
  const hasDeltaSummary = deltaSummary !== null;

  const modelImportanceRows = (snapshot.model_diagnostics?.win_model?.feature_importance ?? [])
    .slice(0, 6)
    .map(
      (row) => `
      <div class="matchup-weight-row">
        <span>${safeText(row.feature ?? "-")}</span>
        <strong>${safeText(formatNumber(row.importance_pct, 1))}%</strong>
      </div>
      `,
    )
    .join("");

  const statRows = makeMatchupStatRows(homeTeam, awayTeam, game)
    .map(
      (row) => `
      <div class="matchup-sbs-row">
        <span class="matchup-sbs-home">${safeText(row.home)}</span>
        <span class="matchup-sbs-label">${safeText(row.label)}</span>
        <span class="matchup-sbs-away">${safeText(row.away)}</span>
      </div>
      `,
    )
    .join("");

  const contributionRows = componentRows
    .map(
      (row) => `
      <div class="matchup-weight-row">
        <span>${safeText(row.label)}</span>
        <strong>${safeText(formatSignedPctPoints(row.value))}</strong>
      </div>
      `,
    )
    .join("");

  const whyRowsData = buildWhyPickRows(game);
  const whyScale = Math.max(0.01, ...whyRowsData.map((row) => Math.abs(Number(row.value ?? 0))));
  const whyRows = whyRowsData
    .map((row) => {
      const raw = Number(row.value ?? 0);
      const width = Math.max(3, (Math.abs(raw) / whyScale) * 100);
      const cls = raw >= 0 ? "why-pick-fill pos" : "why-pick-fill neg";
      return `
      <div class="why-pick-row">
        <div class="why-pick-meta">
          <span>${safeText(row.label)}</span>
          <strong class="${raw >= 0 ? "delta-up" : "delta-down"}">${safeText(formatSignedPctPoints(raw))}</strong>
        </div>
        <div class="why-pick-track">
          <div class="${cls}" style="width:${width.toFixed(1)}%"></div>
        </div>
      </div>
      `;
    })
    .join("");

  const deltaRows = deltaRowsData.rows
    .map((row) => {
      const value = Number(row.value ?? 0);
      return `
      <div class="matchup-weight-row">
        <span>${safeText(row.label)}</span>
        <strong class="${value >= 0 ? "delta-up" : "delta-down"}">${safeText(formatSignedPctPoints(value))}</strong>
      </div>
      `;
    })
    .join("");

  const advancedWaterfallRowsData = buildAdvancedWaterfallRows(game);
  const advancedScale = Math.max(
    0.01,
    ...advancedWaterfallRowsData.flatMap((row) => [Math.abs(Number(row.start ?? 0)), Math.abs(Number(row.end ?? 0)), Math.abs(Number(row.value ?? 0))]),
  );
  const advancedWaterfallRows = advancedWaterfallRowsData
    .map((row) => {
      const startVal = Number(row.start ?? 0);
      const endVal = Number(row.end ?? 0);
      const startPct = 50 + ((startVal / advancedScale) * 45);
      const endPct = 50 + ((endVal / advancedScale) * 45);
      const left = Math.max(1, Math.min(startPct, endPct));
      const width = Math.max(2, Math.abs(endPct - startPct));
      const value = Number(row.value ?? 0);
      const cls = value >= 0 ? "advanced-waterfall-fill pos" : "advanced-waterfall-fill neg";
      return `
      <div class="advanced-waterfall-row">
        <div class="advanced-waterfall-meta">
          <span>${safeText(row.label)}</span>
          <strong class="${value >= 0 ? "delta-up" : "delta-down"}">${safeText(formatSignedPctPoints(value))}</strong>
        </div>
        <div class="advanced-waterfall-track">
          <div class="${cls}" style="left:${left.toFixed(2)}%;width:${width.toFixed(2)}%"></div>
        </div>
        <p class="advanced-waterfall-cum">Cumulative: ${safeText(formatSignedPctPoints(row.cumulative))}</p>
      </div>
      `;
    })
    .join("");

  const deltaMetaParts = [];
  if (deltaRowsData.sourceDate) {
    deltaMetaParts.push(`Source ${deltaRowsData.sourceDate}`);
  }
  if (Number.isFinite(deltaRowsData.days)) {
    deltaMetaParts.push(`${deltaRowsData.days} day lookback`);
  }
  if (deltaRowsData.favoredChanged) {
    deltaMetaParts.push("Favored team changed");
  }

  matchupSidebarTitle.textContent = `${game.away_team} @ ${game.home_team}`;
  matchupSidebarSubtitle.textContent = `${favoredTeam} favored ${formatPercent(favoredProb, 1)} | click another matchup to compare.`;

  matchupSidebarBody.innerHTML = `
    <article class="matchup-summary-card">
      <p><strong>Model Home:</strong> ${safeText(formatPercent(modelHome, 1))}</p>
      <p><strong>Context Adj:</strong> ${safeText(formatSignedPctPoints(contextAdj))}</p>
      <p><strong>Pre-Market Home:</strong> ${safeText(formatPercent(preMarketHome, 1))}</p>
      <p><strong>Market Impact:</strong> ${safeText(formatSignedPctPoints(marketImpact))}</p>
      <p><strong>Final Home:</strong> ${safeText(formatPercent(finalHome, 1))}</p>
      ${hasDeltaSummary ? `<p><strong>Delta Final:</strong> <span class="${deltaSummary >= 0 ? "delta-up" : "delta-down"}">${safeText(formatSignedPctPoints(deltaSummary))}</span>${deltaRowsData.sourceDate ? ` <span class="muted-note">vs ${safeText(deltaRowsData.sourceDate)}</span>` : ""}</p>` : ""}
      <p class="muted-note">${safeText(game.away_probable_pitcher ?? "TBD")} vs ${safeText(game.home_probable_pitcher ?? "TBD")}</p>
    </article>

    <section class="matchup-section">
      <h4>Why This Pick</h4>
      <div class="why-pick-list">${whyRows || "<p class='muted-note'>No contribution breakdown available.</p>"}</div>
    </section>

    <section class="matchup-section">
      <h4>What Changed${deltaRowsData.sourceDate ? ` (vs ${safeText(deltaRowsData.sourceDate)})` : ""}</h4>
      <p class="muted-note">${safeText(deltaMetaParts.length ? deltaMetaParts.join(" | ") : "No archived prior matchup values available yet.")}</p>
      <div class="matchup-weight-list">${deltaRows || "<p class='muted-note'>No delta rows available for this matchup yet.</p>"}</div>
    </section>

    <section class="matchup-section">
      <h4>Advanced Edge Waterfall</h4>
      <div class="advanced-waterfall-list">${advancedWaterfallRows || "<p class='muted-note'>No advanced edge data available.</p>"}</div>
    </section>

    <section class="matchup-section">
      <h4>Side-by-Side Team Context</h4>
      <div class="matchup-sbs-head">
        <span>${safeText(game.home_team ?? "Home")}</span>
        <span>Metric</span>
        <span>${safeText(game.away_team ?? "Away")}</span>
      </div>
      <div class="matchup-sbs-grid">${statRows}</div>
    </section>

    <section class="matchup-section">
      <h4>Game-Specific Model Contributions</h4>
      <div class="matchup-weight-list">${contributionRows}</div>
    </section>

    <section class="matchup-section">
      <h4>Global Model Feature Weights</h4>
      <div class="matchup-weight-list">${modelImportanceRows || "<p class='muted-note'>No feature importance available.</p>"}</div>
    </section>
  `;

  matchupSidebar.classList.remove("is-closed");
  matchupSidebar.setAttribute("aria-hidden", "false");
  if (matchupExportPngBtn) {
    matchupExportPngBtn.disabled = false;
  }
}

function sanitizeFilePart(value) {
  const slug = String(value ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40);
  return slug || "team";
}

function compactDateStamp(dateObj = new Date()) {
  const d = dateObj instanceof Date ? dateObj : new Date(dateObj);
  if (Number.isNaN(d.getTime())) {
    return "unknown";
  }
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mi = String(d.getMinutes()).padStart(2, "0");
  return `${yyyy}${mm}${dd}-${hh}${mi}`;
}

function truncateCanvasText(ctx, text, maxWidth) {
  const input = String(text ?? "");
  if (ctx.measureText(input).width <= maxWidth) {
    return input;
  }

  let out = input;
  while (out.length > 1 && ctx.measureText(`${out}...`).width > maxWidth) {
    out = out.slice(0, -1);
  }
  return `${out}...`;
}

function drawWrappedCanvasText(ctx, text, x, y, maxWidth, lineHeight, maxLines = 3) {
  const words = String(text ?? "").split(/\s+/).filter(Boolean);
  if (!words.length) {
    return y;
  }

  let line = "";
  let linesDrawn = 0;
  let currentY = y;

  for (let i = 0; i < words.length; i += 1) {
    const word = words[i];
    const next = line ? `${line} ${word}` : word;
    if (ctx.measureText(next).width <= maxWidth) {
      line = next;
      continue;
    }

    if (line) {
      ctx.fillText(line, x, currentY);
      linesDrawn += 1;
      currentY += lineHeight;
    }
    line = word;

    if (linesDrawn >= maxLines - 1) {
      break;
    }
  }

  if (linesDrawn < maxLines && line) {
    const remainingText = truncateCanvasText(ctx, line, maxWidth);
    ctx.fillText(remainingText, x, currentY);
    currentY += lineHeight;
  }

  return currentY;
}

function resolveRrgMetricKey(metricRaw) {
  const metric = String(metricRaw || "power").trim().toLowerCase();
  const alias = {
    power: "live_power_score",
    mscore: "mscore",
    talent: "talent_score",
    form: "form_score",
    risk: "risk_score",
    dplus: "d_plus_score",
    d_plus: "d_plus_score",
    "d+": "d_plus_score",
    joe: "joe_score",
    joe_score: "joe_score",
    leverage: "leverage_net_quality",
    clutch: "clutch_index",
    gap: "power_minus_d_plus",
    power_minus: "power_minus_d_plus",
    power_minus_d_plus: "power_minus_d_plus",
  };
  return alias[metric] || "live_power_score";
}

function getRrgParamsFromState(metricOverride = null) {
  const metric = metricOverride || state.rrgMetric || "power";
  const lookback = Math.max(30, Math.min(365, Number(state.rrgLookback || 120)));
  const trail = Math.max(3, Math.min(45, Number(state.rrgTrail || 7)));
  const minPoints = Math.max(4, Math.min(60, Math.floor(trail / 2) + 2));

  const params = new URLSearchParams({
    metric,
    lookback_days: String(lookback),
    trail_days: String(trail),
    min_points: String(minPoints),
  });

  return { metric, params };
}

async function fetchRrgPayloadForExport() {
  const { metric, params } = getRrgParamsFromState();
  const expectedKey = resolveRrgMetricKey(metric);

  if (state.rrgPayload && String(state.rrgPayload.metric_key || "") === expectedKey) {
    return state.rrgPayload;
  }

  try {
    const response = await fetch(`/api/rrg?${params.toString()}`);
    if (!response.ok) {
      throw new Error(`RRG request failed (${response.status})`);
    }
    const payload = await response.json();
    state.rrgPayload = payload;
    return payload;
  } catch (_error) {
    return state.rrgPayload || null;
  }
}

function drawMatchupRrgPanel(ctx, x, y, width, height, payload, game) {
  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#c8d9e8";
  ctx.lineWidth = 3;
  ctx.fillRect(x, y, width, height);
  ctx.strokeRect(x, y, width, height);

  ctx.fillStyle = "#1a3f5a";
  ctx.font = "700 36px 'Segoe UI', Arial, sans-serif";
  const metricLabel = payload?.metric_key ? String(payload.metric_key).replaceAll("_", " ") : "selected metric";
  ctx.fillText(`RRG Snapshot (${metricLabel})`, x + 30, y + 46);

  const homeTeam = String(game?.home_team || "");
  const awayTeam = String(game?.away_team || "");
  const allPoints = payload?.points || [];
  const points = allPoints.filter((row) => {
    const name = String(row?.team || "");
    return name === homeTeam || name === awayTeam;
  });

  if (!points.length) {
    ctx.fillStyle = "#4d6477";
    ctx.font = "500 24px 'Segoe UI', Arial, sans-serif";
    ctx.fillText("RRG data unavailable for this matchup in current lookback window.", x + 30, y + 100);
    return;
  }

  const chartPad = { left: 86, right: 30, top: 72, bottom: 92 };
  const chartX = x + chartPad.left;
  const chartY = y + chartPad.top;
  const chartW = width - chartPad.left - chartPad.right;
  const chartH = height - chartPad.top - chartPad.bottom;

  const valuesX = [];
  const valuesY = [];
  for (const row of points) {
    const px = Number(row?.x);
    const py = Number(row?.y);
    if (Number.isFinite(px)) valuesX.push(px);
    if (Number.isFinite(py)) valuesY.push(py);
    const trail = row?.trail || [];
    for (const trow of trail) {
      const tx = Number(trow?.x);
      const ty = Number(trow?.y);
      if (Number.isFinite(tx)) valuesX.push(tx);
      if (Number.isFinite(ty)) valuesY.push(ty);
    }
  }

  const span = Math.max(
    2,
    ...valuesX.map((v) => Math.abs(v - 100)),
    ...valuesY.map((v) => Math.abs(v - 100)),
  );
  const halfSpan = Math.ceil((span + 0.25) * 2) / 2;
  const xMin = 100 - halfSpan;
  const xMax = 100 + halfSpan;
  const yMin = 100 - halfSpan;
  const yMax = 100 + halfSpan;

  const sx = (val) => chartX + ((Number(val) - xMin) / Math.max(1e-9, xMax - xMin)) * chartW;
  const sy = (val) => chartY + ((yMax - Number(val)) / Math.max(1e-9, yMax - yMin)) * chartH;

  const x100 = sx(100);
  const y100 = sy(100);

  const rects = [
    { c: "rgba(15,143,99,0.10)", x: x100, y: chartY, w: chartX + chartW - x100, h: y100 - chartY },
    { c: "rgba(255,154,74,0.12)", x: x100, y: y100, w: chartX + chartW - x100, h: chartY + chartH - y100 },
    { c: "rgba(120,132,145,0.12)", x: chartX, y: y100, w: x100 - chartX, h: chartY + chartH - y100 },
    { c: "rgba(38,121,210,0.11)", x: chartX, y: chartY, w: x100 - chartX, h: y100 - chartY },
  ];
  for (const r of rects) {
    ctx.fillStyle = r.c;
    ctx.fillRect(r.x, r.y, Math.max(0, r.w), Math.max(0, r.h));
  }

  ctx.strokeStyle = "#2f4254";
  ctx.lineWidth = 2;
  ctx.setLineDash([7, 6]);
  ctx.beginPath();
  ctx.moveTo(chartX, y100);
  ctx.lineTo(chartX + chartW, y100);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x100, chartY);
  ctx.lineTo(x100, chartY + chartH);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = "#9eb4c8";
  ctx.lineWidth = 2;
  ctx.strokeRect(chartX, chartY, chartW, chartH);

  const colorByTeam = {
    [homeTeam]: "#2563eb",
    [awayTeam]: "#d97706",
  };

  for (const row of points) {
    const team = String(row?.team || "");
    const color = colorByTeam[team] || "#1a9aaf";
    const trail = row?.trail || [];

    if (trail.length >= 2) {
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.45;
      ctx.lineWidth = 3;
      ctx.beginPath();
      for (let i = 0; i < trail.length; i += 1) {
        const tx = sx(trail[i].x);
        const ty = sy(trail[i].y);
        if (i === 0) {
          ctx.moveTo(tx, ty);
        } else {
          ctx.lineTo(tx, ty);
        }
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    const px = sx(row?.x);
    const py = sy(row?.y);
    ctx.fillStyle = color;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px, py, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#1f3345";
    ctx.font = "700 21px 'Segoe UI', Arial, sans-serif";
    const label = shortTeamName(team);
    ctx.fillText(label, px + 12, py - 8);
  }

  ctx.fillStyle = "#52657a";
  ctx.font = "500 18px 'Segoe UI', Arial, sans-serif";
  const step = Math.max(1, Math.ceil((xMax - xMin) / 8));
  for (let v = Math.ceil(xMin); v <= Math.floor(xMax); v += step) {
    const tx = sx(v);
    ctx.fillText(String(v), tx - 12, chartY + chartH + 28);
  }
  for (let v = Math.ceil(yMin); v <= Math.floor(yMax); v += step) {
    const ty = sy(v);
    ctx.fillText(String(v), chartX - 44, ty + 6);
  }

  ctx.fillStyle = "#33516b";
  ctx.font = "600 20px 'Segoe UI', Arial, sans-serif";
  ctx.fillText("RS-Ratio", chartX + chartW / 2 - 45, chartY + chartH + 60);

  ctx.save();
  ctx.translate(chartX - 62, chartY + chartH / 2 + 46);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("RS-Momentum", 0, 0);
  ctx.restore();

  const legendY = y + height - 36;
  const legend = [
    { name: homeTeam || "Home", color: colorByTeam[homeTeam] || "#2563eb" },
    { name: awayTeam || "Away", color: colorByTeam[awayTeam] || "#d97706" },
  ];
  let legendX = x + 30;
  for (const item of legend) {
    ctx.fillStyle = item.color;
    ctx.fillRect(legendX, legendY - 14, 18, 18);
    ctx.fillStyle = "#2b465f";
    ctx.font = "600 20px 'Segoe UI', Arial, sans-serif";
    const name = truncateCanvasText(ctx, item.name, 380);
    ctx.fillText(name, legendX + 26, legendY);
    legendX += 420;
  }
}

async function exportMatchupBreakdownPng(snapshot) {
  if (!snapshot || !state.selectedMatchupId) {
    return;
  }

  const game = findMatchupById(snapshot, state.selectedMatchupId);
  if (!game) {
    return;
  }

  const homeTeam = findTeam(snapshot, game.home_team);
  const awayTeam = findTeam(snapshot, game.away_team);
  const componentRows = buildMatchupComponentRows(game);
  const advancedWaterfallRows = buildAdvancedWaterfallRows(game);
  const statRows = makeMatchupStatRows(homeTeam, awayTeam, game);
  const modelWeightRows = (snapshot.model_diagnostics?.win_model?.feature_importance ?? []).slice(0, 8);

  const contextAdj = componentRows.reduce((sum, row) => sum + Number(row.value ?? 0), 0);
  const modelHome = Number(game.model_home_win_prob ?? game.home_win_prob ?? 0.5);
  const preMarketHome = Math.max(0.02, Math.min(0.98, modelHome + contextAdj));
  const finalHome = Number(game.home_win_prob ?? preMarketHome);
  const marketImpact = finalHome - preMarketHome;
  const favoredTeam = game.favored_team ?? game.home_team;
  const favoredProb = Number(game.favored_win_prob ?? 0.5);
  const rrgPayload = await fetchRrgPayloadForExport();

  // 8.5 x 11 inches at 300 DPI.
  const pageWidth = 2550;
  const pageHeight = 3300;
  const margin = 140;
  const contentWidth = pageWidth - margin * 2;

  const canvas = document.createElement("canvas");
  canvas.width = pageWidth;
  canvas.height = pageHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, pageWidth, pageHeight);

  let y = margin;

  // Header band.
  ctx.fillStyle = "#12324a";
  ctx.fillRect(margin, y, contentWidth, 210);
  ctx.fillStyle = "#ffffff";
  ctx.font = "700 72px 'Segoe UI', Arial, sans-serif";
  ctx.fillText("MLB Matchup Breakdown", margin + 48, y + 88);
  ctx.font = "600 52px 'Segoe UI', Arial, sans-serif";
  ctx.fillText(`${game.away_team ?? "Away"} @ ${game.home_team ?? "Home"}`, margin + 48, y + 156);
  y += 250;

  const gameDate = game.official_date ? new Date(`${game.official_date}T00:00:00`).toLocaleDateString() : "N/A";
  const generatedAt = snapshot?.meta?.generated_at ? new Date(snapshot.meta.generated_at).toLocaleString() : "N/A";

  ctx.fillStyle = "#2a4963";
  ctx.font = "500 31px 'Segoe UI', Arial, sans-serif";
  ctx.fillText(`Game Date: ${gameDate}`, margin, y);
  ctx.fillText(`Generated: ${generatedAt}`, margin + 920, y);
  y += 52;

  // Summary card.
  const summaryTop = y;
  const summaryHeight = 280;
  ctx.fillStyle = "#f8fbff";
  ctx.strokeStyle = "#c8d9e8";
  ctx.lineWidth = 3;
  ctx.fillRect(margin, summaryTop, contentWidth, summaryHeight);
  ctx.strokeRect(margin, summaryTop, contentWidth, summaryHeight);

  ctx.fillStyle = "#1a3f5a";
  ctx.font = "700 38px 'Segoe UI', Arial, sans-serif";
  ctx.fillText("Game Probability Summary", margin + 30, summaryTop + 52);

  const summaryPairs = [
    ["Favored", `${favoredTeam} (${formatPercent(favoredProb, 1)})`],
    ["Model Home", formatPercent(modelHome, 1)],
    ["Context Adj", formatSignedPctPoints(contextAdj)],
    ["Pre-Market Home", formatPercent(preMarketHome, 1)],
    ["Market Impact", formatSignedPctPoints(marketImpact)],
    ["Final Home", formatPercent(finalHome, 1)],
  ];

  const summaryCols = 2;
  const summaryRows = Math.ceil(summaryPairs.length / summaryCols);
  const colWidth = (contentWidth - 60) / summaryCols;

  for (let i = 0; i < summaryPairs.length; i += 1) {
    const row = Math.floor(i / summaryCols);
    const col = i % summaryCols;
    const sx = margin + 30 + col * colWidth;
    const sy = summaryTop + 92 + row * 56;
    ctx.fillStyle = "#45627b";
    ctx.font = "600 27px 'Segoe UI', Arial, sans-serif";
    ctx.fillText(summaryPairs[i][0], sx, sy);
    ctx.fillStyle = "#0f2940";
    ctx.font = "700 30px 'Segoe UI', Arial, sans-serif";
    ctx.fillText(truncateCanvasText(ctx, summaryPairs[i][1], colWidth - 30), sx + 190, sy);
  }

  ctx.fillStyle = "#4a657d";
  ctx.font = "500 25px 'Segoe UI', Arial, sans-serif";
  drawWrappedCanvasText(
    ctx,
    `${game.away_probable_pitcher ?? "TBD"} vs ${game.home_probable_pitcher ?? "TBD"}`,
    margin + 30,
    summaryTop + summaryHeight - 46,
    contentWidth - 60,
    30,
    1,
  );
  y += summaryHeight + 34;

  // Side-by-side stats table.
  const tableTop = y;
  const tableHeaderH = 52;
  const tableRowH = 44;
  const tableHeight = 64 + tableHeaderH + statRows.length * tableRowH;

  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#c8d9e8";
  ctx.lineWidth = 3;
  ctx.fillRect(margin, tableTop, contentWidth, tableHeight);
  ctx.strokeRect(margin, tableTop, contentWidth, tableHeight);

  ctx.fillStyle = "#1a3f5a";
  ctx.font = "700 36px 'Segoe UI', Arial, sans-serif";
  ctx.fillText("Side-by-Side Team Context", margin + 30, tableTop + 46);

  const tableY = tableTop + 64;
  const colHomeW = contentWidth * 0.30;
  const colMetricW = contentWidth * 0.40;
  const colAwayW = contentWidth - colHomeW - colMetricW;

  ctx.fillStyle = "#e8f1f9";
  ctx.fillRect(margin + 2, tableY, contentWidth - 4, tableHeaderH);

  ctx.fillStyle = "#1c3f5b";
  ctx.font = "700 26px 'Segoe UI', Arial, sans-serif";
  ctx.fillText(truncateCanvasText(ctx, game.home_team ?? "Home", colHomeW - 28), margin + 16, tableY + 34);
  ctx.fillText("Metric", margin + colHomeW + 16, tableY + 34);
  ctx.fillText(truncateCanvasText(ctx, game.away_team ?? "Away", colAwayW - 28), margin + colHomeW + colMetricW + 16, tableY + 34);

  for (let i = 0; i < statRows.length; i += 1) {
    const row = statRows[i];
    const ry = tableY + tableHeaderH + i * tableRowH;
    if (i % 2 === 0) {
      ctx.fillStyle = "#f8fbff";
      ctx.fillRect(margin + 2, ry, contentWidth - 4, tableRowH);
    }

    ctx.strokeStyle = "#e0ecf5";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(margin + 2, ry + tableRowH);
    ctx.lineTo(margin + contentWidth - 2, ry + tableRowH);
    ctx.stroke();

    ctx.fillStyle = "#23455f";
    ctx.font = "600 24px 'Segoe UI', Arial, sans-serif";
    ctx.fillText(truncateCanvasText(ctx, row.home, colHomeW - 28), margin + 16, ry + 30);

    ctx.fillStyle = "#4d6477";
    ctx.font = "500 23px 'Segoe UI', Arial, sans-serif";
    ctx.fillText(truncateCanvasText(ctx, row.label, colMetricW - 28), margin + colHomeW + 16, ry + 30);

    ctx.fillStyle = "#23455f";
    ctx.font = "600 24px 'Segoe UI', Arial, sans-serif";
    ctx.fillText(truncateCanvasText(ctx, row.away, colAwayW - 28), margin + colHomeW + colMetricW + 16, ry + 30);
  }

  y += tableHeight + 34;

  // Matchup-only RRG panel (both teams only).
  const rrgPanelH = 520;
  drawMatchupRrgPanel(ctx, margin, y, contentWidth, rrgPanelH, rrgPayload, game);
  y += rrgPanelH + 34;

  // Lower two-column cards.
  const cardGap = 26;
  const cardW = (contentWidth - cardGap) / 2;
  const leftX = margin;
  const rightX = margin + cardW + cardGap;
  const lowerTop = y;

  const contribCardH = 122 + componentRows.length * 44 + Math.max(1, advancedWaterfallRows.length) * 34;
  const modelCardH = 86 + Math.max(1, modelWeightRows.length) * 44;
  const lowerH = Math.max(contribCardH, modelCardH);

  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#c8d9e8";
  ctx.lineWidth = 3;
  ctx.fillRect(leftX, lowerTop, cardW, lowerH);
  ctx.strokeRect(leftX, lowerTop, cardW, lowerH);
  ctx.fillRect(rightX, lowerTop, cardW, lowerH);
  ctx.strokeRect(rightX, lowerTop, cardW, lowerH);

  ctx.fillStyle = "#1a3f5a";
  ctx.font = "700 31px 'Segoe UI', Arial, sans-serif";
  ctx.fillText("Game-Specific + Advanced Edge", leftX + 24, lowerTop + 42);
  ctx.fillText("Global Model Weights", rightX + 24, lowerTop + 42);

  for (let i = 0; i < componentRows.length; i += 1) {
    const row = componentRows[i];
    const ry = lowerTop + 76 + i * 44;
    ctx.fillStyle = "#4d6477";
    ctx.font = "500 23px 'Segoe UI', Arial, sans-serif";
    ctx.fillText(truncateCanvasText(ctx, row.label, cardW - 240), leftX + 24, ry);
    ctx.fillStyle = "#173b57";
    ctx.font = "700 24px 'Segoe UI', Arial, sans-serif";
    ctx.fillText(formatSignedPctPoints(row.value), leftX + cardW - 190, ry);
  }

  const advTitleY = lowerTop + 76 + componentRows.length * 44 + 18;
  ctx.fillStyle = "#2a4d6a";
  ctx.font = "700 24px 'Segoe UI', Arial, sans-serif";
  ctx.fillText("Advanced Edge Waterfall", leftX + 24, advTitleY);

  for (let i = 0; i < advancedWaterfallRows.length; i += 1) {
    const row = advancedWaterfallRows[i];
    const ry = advTitleY + 26 + i * 34;
    ctx.fillStyle = "#4d6477";
    ctx.font = "500 20px 'Segoe UI', Arial, sans-serif";
    ctx.fillText(truncateCanvasText(ctx, row.label, cardW - 280), leftX + 24, ry);
    ctx.fillStyle = "#173b57";
    ctx.font = "700 20px 'Segoe UI', Arial, sans-serif";
    ctx.fillText(formatSignedPctPoints(row.value), leftX + cardW - 250, ry);
    ctx.fillStyle = "#6a8093";
    ctx.font = "500 18px 'Segoe UI', Arial, sans-serif";
    ctx.fillText(`Cum ${formatSignedPctPoints(row.cumulative)}`, leftX + cardW - 132, ry);
  }

  if (!modelWeightRows.length) {
    ctx.fillStyle = "#4d6477";
    ctx.font = "500 22px 'Segoe UI', Arial, sans-serif";
    ctx.fillText("No feature-importance data available.", rightX + 24, lowerTop + 104);
  } else {
    for (let i = 0; i < modelWeightRows.length; i += 1) {
      const row = modelWeightRows[i];
      const ry = lowerTop + 76 + i * 44;
      ctx.fillStyle = "#4d6477";
      ctx.font = "500 23px 'Segoe UI', Arial, sans-serif";
      ctx.fillText(truncateCanvasText(ctx, row.feature ?? "-", cardW - 240), rightX + 24, ry);
      ctx.fillStyle = "#173b57";
      ctx.font = "700 24px 'Segoe UI', Arial, sans-serif";
      ctx.fillText(`${formatNumber(row.importance_pct, 1)}%`, rightX + cardW - 170, ry);
    }
  }

  // Footer
  ctx.fillStyle = "#6a8093";
  ctx.font = "500 22px 'Segoe UI', Arial, sans-serif";
  ctx.fillText("Formatted for 8.5 x 11 portrait print. Source: MLB Stats API + model diagnostics.", margin, pageHeight - margin + 26);

  const awaySlug = sanitizeFilePart(game.away_team);
  const homeSlug = sanitizeFilePart(game.home_team);
  const dateSlug = String(game.official_date || new Date().toISOString().slice(0, 10));
  const filename = `matchup-breakdown-${awaySlug}-at-${homeSlug}-${dateSlug}.png`;

  canvas.toBlob((blob) => {
    if (!blob) {
      return;
    }
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }, "image/png");
}

function shortTeamName(teamName) {
  const parts = String(teamName ?? "").trim().split(/\s+/).filter(Boolean);
  if (!parts.length) {
    return "";
  }
  if (parts.length === 1) {
    return parts[0].slice(0, 3).toUpperCase();
  }
  return parts.map((part) => part[0]).join("").slice(0, 3).toUpperCase();
}

function updateRrgLabels() {
  if (rrgLookbackLabel && rrgLookback) {
    rrgLookbackLabel.textContent = `${rrgLookback.value}d`;
  }
}

function setRrgTeamMenuOpen(isOpen) {
  state.rrgTeamMenuOpen = Boolean(isOpen);
  if (!rrgTeamMenu) {
    return;
  }
  rrgTeamMenu.classList.toggle("is-closed", !state.rrgTeamMenuOpen);
}

function syncRrgTeamSelection(payload) {
  const teams = Array.from(new Set((payload?.points ?? []).map((row) => String(row.team ?? "")).filter(Boolean))).sort();
  const previousUniverse = state.rrgTeamUniverse ?? [];
  const previousSelection = new Set(state.rrgSelectedTeams ?? []);
  const hadAllSelected = previousUniverse.length > 0 && previousSelection.size === previousUniverse.length;

  state.rrgTeamUniverse = teams;

  if (!state.rrgTeamSelectionInitialized) {
    if (!teams.length) {
      return;
    }
    state.rrgSelectedTeams = [...teams];
    state.rrgTeamSelectionInitialized = true;
    return;
  }

  if (hadAllSelected) {
    state.rrgSelectedTeams = [...teams];
    return;
  }

  state.rrgSelectedTeams = teams.filter((team) => previousSelection.has(team));
}

function renderRrgTeamSelector() {
  if (!rrgTeamOptions || !rrgTeamToggle) {
    return;
  }

  const teams = state.rrgTeamUniverse ?? [];
  const selected = new Set(state.rrgSelectedTeams ?? []);

  if (!teams.length) {
    rrgTeamOptions.innerHTML = `<p class="muted-note">No teams available.</p>`;
    rrgTeamToggle.textContent = "Teams: N/A";
    return;
  }

  const selectedCount = teams.filter((team) => selected.has(team)).length;
  if (selectedCount === 0) {
    rrgTeamToggle.textContent = "Teams: None";
  } else if (selectedCount === teams.length) {
    rrgTeamToggle.textContent = `Teams: All (${teams.length})`;
  } else {
    rrgTeamToggle.textContent = `Teams: ${selectedCount}/${teams.length}`;
  }

  rrgTeamOptions.innerHTML = teams
    .map((team) => {
      const checked = selected.has(team) ? "checked" : "";
      return `
        <label class="rrg-team-option">
          <input type="checkbox" data-team="${safeText(team)}" ${checked} />
          <span>${safeText(team)}</span>
        </label>
      `;
    })
    .join("");
}

function applyRrgChartSize() {
  if (!rrgSvg || !rrgChartWrap) {
    return;
  }

  const size = Math.max(100, Math.min(200, Number(state.rrgSize || 100)));
  rrgSvg.style.width = `${size}%`;
  rrgSvg.style.maxWidth = "none";
  rrgSvg.style.height = "auto";
  rrgChartWrap.style.overflowX = size > 100 ? "auto" : "hidden";
}

function parseOptionalNumber(value) {
  const raw = String(value ?? "").trim();
  if (!raw) {
    return null;
  }
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : null;
}

function rrgMetricDisplayName(metricRaw) {
  const key = resolveRrgMetricKey(metricRaw);
  const labels = {
    live_power_score: "Power Score",
    d_plus_score: "D+ Score",
    joe_score: "Joe Score",
  };
  return labels[key] || String(key || "Metric").replaceAll("_", " ");
}

function buildMomentumLeaders(points, direction) {
  const rows = (points ?? [])
    .map((row) => {
      const momentum = Number(row?.y);
      if (!Number.isFinite(momentum)) {
        return null;
      }
      return {
        team: String(row?.team ?? ""),
        momentum,
        delta: momentum - 100,
      };
    })
    .filter(Boolean)
    .filter((row) => row.team);

  rows.sort((a, b) => {
    if (direction === "improving") {
      if (b.momentum !== a.momentum) return b.momentum - a.momentum;
      return a.team.localeCompare(b.team);
    }
    if (a.momentum !== b.momentum) return a.momentum - b.momentum;
    return a.team.localeCompare(b.team);
  });

  return rows.slice(0, 5);
}

function renderMomentumLeaderList(rows) {
  if (!rows.length) {
    return `<p class="rrg-momentum-empty">No teams available.</p>`;
  }

  return `
    <ol class="rrg-momentum-list">
      ${rows
        .map((row) => {
          const deltaClass = row.delta >= 0 ? "up" : "down";
          const deltaText = row.delta >= 0 ? `+${row.delta.toFixed(2)}` : row.delta.toFixed(2);
          return `
            <li class="rrg-momentum-item">
              <span class="rrg-momentum-team">${safeText(row.team)}</span>
              <span class="rrg-momentum-value">${row.momentum.toFixed(2)}<span class="rrg-momentum-delta ${deltaClass}">${safeText(deltaText)}</span></span>
            </li>
          `;
        })
        .join("")}
    </ol>
  `;
}

function renderRrgMomentumLeaders(boards, options = {}) {
  if (!rrgMomentumBoards || !rrgMomentumNote) {
    return;
  }

  const validBoards = Array.isArray(boards) ? boards.filter((row) => row && !row.error) : [];
  if (!validBoards.length) {
    state.rrgMomentumLeaders = [];
    state.rrgMomentumMeta = null;
    rrgMomentumBoards.innerHTML = `<p class="muted-note">Momentum leaders are unavailable right now.</p>`;
    rrgMomentumNote.textContent = options.message || "Could not load momentum leader data.";
    if (rrgMomentumExportPngBtn) {
      rrgMomentumExportPngBtn.disabled = true;
    }
    return;
  }

  const preparedBoards = validBoards.map((board) => {
    const title = rrgMetricDisplayName(board.metric);
    const improving = buildMomentumLeaders(board.points, "improving");
    const regressing = buildMomentumLeaders(board.points, "regressing");
    return {
      metric: board.metric,
      metricKey: resolveRrgMetricKey(board.metric),
      title,
      improving,
      regressing,
    };
  });

  state.rrgMomentumLeaders = preparedBoards;
  state.rrgMomentumMeta = {
    lookback: Number(options.lookback ?? state.rrgLookback ?? 120),
    trail: Number(options.trail ?? state.rrgTrail ?? 7),
    generatedAt: new Date().toISOString(),
  };

  rrgMomentumBoards.innerHTML = preparedBoards
    .map((board) => `
        <article class="rrg-momentum-card">
          <h4>${safeText(board.title)}</h4>
          <div class="rrg-momentum-columns">
            <div class="rrg-momentum-col">
              <h5>Most Improving</h5>
              ${renderMomentumLeaderList(board.improving)}
            </div>
            <div class="rrg-momentum-col">
              <h5>Most Regressing</h5>
              ${renderMomentumLeaderList(board.regressing)}
            </div>
          </div>
        </article>
      `)
    .join("");

  const lookback = state.rrgMomentumMeta.lookback;
  const trail = state.rrgMomentumMeta.trail;
  rrgMomentumNote.textContent = `Based on RS-Momentum with ${lookback}d lookback and ${trail}d trail settings.`;
  if (rrgMomentumExportPngBtn) {
    rrgMomentumExportPngBtn.disabled = false;
  }
}

async function loadRrgMomentumLeaders(primaryPayload = null, primaryMetric = null) {
  if (!rrgMomentumBoards || !rrgMomentumNote) {
    return;
  }

  const targets = ["power", "dplus", "joe"];
  const payloadByMetric = {};
  if (primaryPayload && targets.includes(String(primaryMetric || ""))) {
    payloadByMetric[String(primaryMetric)] = primaryPayload;
  }

  const fetchTargets = targets.filter((metric) => !payloadByMetric[metric]);

  const results = await Promise.allSettled(
    fetchTargets.map(async (metric) => {
      const { params } = getRrgParamsFromState(metric);
      const response = await fetch(`/api/rrg?${params.toString()}`);
      if (!response.ok) {
        throw new Error(`RRG request failed (${response.status})`);
      }
      const payload = await response.json();
      return { metric, payload };
    }),
  );

  const boards = [];
  for (const metric of targets) {
    const existing = payloadByMetric[metric];
    if (existing) {
      boards.push({ metric, points: existing.points ?? [] });
      continue;
    }
    const matched = results.find((entry) => entry.status === "fulfilled" && entry.value?.metric === metric);
    if (matched && matched.status === "fulfilled") {
      boards.push({ metric, points: matched.value.payload?.points ?? [] });
    }
  }

  renderRrgMomentumLeaders(boards, {
    lookback: Number(state.rrgLookback || 120),
    trail: Number(state.rrgTrail || 7),
  });
}

function renderRrg(payload) {
  if (!rrgSvg || !rrgLegend || !rrgValidation) {
    return;
  }

  applyRrgChartSize();

  const pointsRaw = payload?.points ?? [];
  const selectedTeams = new Set(state.rrgSelectedTeams ?? []);
  const points = pointsRaw.filter((row) => selectedTeams.has(String(row.team ?? "")));

  if (!points.length) {
    rrgSvg.innerHTML = `<text x="24" y="42" fill="#52657a" font-size="16">No RRG data for current filters.</text>`;
    rrgLegend.innerHTML = "";
    if ((state.rrgSelectedTeams ?? []).length === 0) {
      rrgValidation.textContent = "No teams selected. Use the Teams dropdown to choose one or more clubs.";
    } else if (payload?.history_note) {
      rrgValidation.textContent = payload.history_note;
    }
    return;
  }

  const viewport = payload?.viewport ?? {};
  const xMinRaw = Number.isFinite(Number(viewport.x_min)) ? Number(viewport.x_min) : 94;
  const xMaxRaw = Number.isFinite(Number(viewport.x_max)) ? Number(viewport.x_max) : 106;
  const yMinRaw = Number.isFinite(Number(viewport.y_min)) ? Number(viewport.y_min) : 94;
  const yMaxRaw = Number.isFinite(Number(viewport.y_max)) ? Number(viewport.y_max) : 106;

  const centeredSpan = Math.max(
    2,
    Math.abs(xMinRaw - 100),
    Math.abs(xMaxRaw - 100),
    Math.abs(yMinRaw - 100),
    Math.abs(yMaxRaw - 100),
  );
  const halfSpan = Math.ceil((centeredSpan + 0.25) * 2) / 2;
  const xMin = 100 - halfSpan;
  const xMax = 100 + halfSpan;
  const yMinAuto = 100 - halfSpan;
  const yMaxAuto = 100 + halfSpan;

  const hasYMinOverride = Number.isFinite(Number(state.rrgYMin));
  const hasYMaxOverride = Number.isFinite(Number(state.rrgYMax));
  let yMin = hasYMinOverride ? Number(state.rrgYMin) : yMinAuto;
  let yMax = hasYMaxOverride ? Number(state.rrgYMax) : yMaxAuto;
  let yOverrideNote = "";

  if ((hasYMinOverride || hasYMaxOverride) && !(yMax > yMin)) {
    yMin = yMinAuto;
    yMax = yMaxAuto;
    yOverrideNote = "Y-axis override ignored (set Y Min < Y Max).";
  }

  if (rrgYMinInput && !hasYMinOverride) {
    rrgYMinInput.placeholder = yMinAuto.toFixed(1);
  }
  if (rrgYMaxInput && !hasYMaxOverride) {
    rrgYMaxInput.placeholder = yMaxAuto.toFixed(1);
  }

  const width = 960;
  const height = 460;
  const pad = { left: 70, right: 28, top: 18, bottom: 48 };
  const fullPlotW = width - pad.left - pad.right;
  const fullPlotH = height - pad.top - pad.bottom;

  // Visual ratio target: X axis area is 50% larger than Y axis area.
  // Data ranges remain symmetric around 100 on each axis.
  const targetAspect = 1.5;
  let plotH = Math.max(10, Math.min(fullPlotH, fullPlotW / targetAspect));
  let plotW = plotH * targetAspect;
  if (plotW > fullPlotW) {
    plotW = fullPlotW;
    plotH = plotW / targetAspect;
  }

  const plotLeft = pad.left + (fullPlotW - plotW) / 2;
  const plotTop = pad.top + (fullPlotH - plotH) / 2;

  const scaleX = (value) => plotLeft + ((Number(value) - xMin) / Math.max(1e-9, xMax - xMin)) * plotW;
  const scaleY = (value) => plotTop + ((yMax - Number(value)) / Math.max(1e-9, yMax - yMin)) * plotH;

  const x100 = Math.max(plotLeft, Math.min(plotLeft + plotW, scaleX(100)));
  const y100 = Math.max(plotTop, Math.min(plotTop + plotH, scaleY(100)));

  const quadrantStyle = {
    Leading: { fill: "rgba(15,143,99,0.10)", stroke: "#0f8f63" },
    Weakening: { fill: "rgba(255,154,74,0.12)", stroke: "#c76f24" },
    Lagging: { fill: "rgba(120,132,145,0.12)", stroke: "#596878" },
    Improving: { fill: "rgba(38,121,210,0.11)", stroke: "#2679d2" },
  };

  const quadrantRects = [
    { key: "Leading", x: x100, y: plotTop, w: plotLeft + plotW - x100, h: y100 - plotTop },
    { key: "Weakening", x: x100, y: y100, w: plotLeft + plotW - x100, h: plotTop + plotH - y100 },
    { key: "Lagging", x: plotLeft, y: y100, w: x100 - plotLeft, h: plotTop + plotH - y100 },
    { key: "Improving", x: plotLeft, y: plotTop, w: x100 - plotLeft, h: y100 - plotTop },
  ];

  const quadrantSvg = quadrantRects
    .map((q) => `<rect x="${q.x.toFixed(1)}" y="${q.y.toFixed(1)}" width="${Math.max(0, q.w).toFixed(1)}" height="${Math.max(0, q.h).toFixed(1)}" fill="${quadrantStyle[q.key].fill}" />`)
    .join("");

  const axisTicksX = [];
  const axisTicksY = [];
  const xStep = Math.max(1, Math.ceil((xMax - xMin) / 10));
  const yStep = Math.max(1, Math.ceil((yMax - yMin) / 10));
  for (let x = Math.ceil(xMin); x <= Math.floor(xMax); x += xStep) {
    const sx = scaleX(x);
    axisTicksX.push(`<line x1="${sx}" y1="${plotTop + plotH}" x2="${sx}" y2="${plotTop + plotH + 5}" stroke="#7a8c9f" stroke-width="1" />`);
    axisTicksX.push(`<text x="${sx}" y="${plotTop + plotH + 20}" text-anchor="middle" fill="#52657a" font-size="11">${x}</text>`);
  }
  for (let y = Math.ceil(yMin); y <= Math.floor(yMax); y += yStep) {
    const sy = scaleY(y);
    axisTicksY.push(`<line x1="${plotLeft - 5}" y1="${sy}" x2="${plotLeft}" y2="${sy}" stroke="#7a8c9f" stroke-width="1" />`);
    axisTicksY.push(`<text x="${plotLeft - 10}" y="${sy + 4}" text-anchor="end" fill="#52657a" font-size="11">${y}</text>`);
  }

  const pointColor = {
    Leading: "#0f8f63",
    Weakening: "#c76f24",
    Lagging: "#596878",
    Improving: "#2679d2",
  };

  const trails = !state.rrgShowTrails
    ? ""
    : points
      .map((row) => {
        const trail = row.trail ?? [];
        if (trail.length < 2) return "";
        const d = trail
          .map((pt, idx) => `${idx === 0 ? "M" : "L"} ${scaleX(Number(pt.x)).toFixed(2)} ${scaleY(Number(pt.y)).toFixed(2)}`)
          .join(" ");
        const color = pointColor[row.quadrant] || "#1a9aaf";
        return `<path d="${d}" fill="none" stroke="${color}" stroke-opacity="0.35" stroke-width="1.9" />`;
      })
      .join("");

  const bubbles = points
    .map((row) => {
      const x = scaleX(Number(row.x));
      const y = scaleY(Number(row.y));
      const color = pointColor[row.quadrant] || "#1a9aaf";
      const teamShort = shortTeamName(row.team);
      const deltaText = row.delta_7d === null || row.delta_7d === undefined ? "n/a" : Number(row.delta_7d).toFixed(2);
      const deltaDays = Number(row.delta_days || 7);
      const metricText = Number(row.metric).toFixed(2);
      return `
        <g>
          <circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="5.4" fill="${color}" fill-opacity="0.92" stroke="#ffffff" stroke-width="1.2">
            <title>${safeText(row.team)} | ${safeText(payload.metric)} ${metricText} | ${deltaDays}d ${deltaText} | ${safeText(row.quadrant)}</title>
          </circle>
          <text x="${(x + 7).toFixed(2)}" y="${(y - 6).toFixed(2)}" fill="#243f58" font-size="10" font-weight="700">${safeText(teamShort)}</text>
        </g>
      `;
    })
    .join("");

  rrgSvg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="#f9fcff" />
    ${quadrantSvg}
    <line x1="${plotLeft}" y1="${y100}" x2="${plotLeft + plotW}" y2="${y100}" stroke="#2f4254" stroke-width="1.3" stroke-dasharray="4 4" />
    <line x1="${x100}" y1="${plotTop}" x2="${x100}" y2="${plotTop + plotH}" stroke="#2f4254" stroke-width="1.3" stroke-dasharray="4 4" />
    <rect x="${plotLeft}" y="${plotTop}" width="${plotW}" height="${plotH}" fill="none" stroke="#9eb4c8" stroke-width="1" />
    ${axisTicksX.join("")}
    ${axisTicksY.join("")}
    ${trails}
    ${bubbles}
    <text x="${width / 2}" y="${height - 12}" text-anchor="middle" fill="#33516b" font-size="12" font-weight="600">RS-Ratio (Relative Strength)</text>
    <text x="18" y="${height / 2}" text-anchor="middle" fill="#33516b" font-size="12" font-weight="600" transform="rotate(-90 18 ${height / 2})">RS-Momentum</text>
    <text x="${(x100 + 8).toFixed(2)}" y="${(plotTop + 16).toFixed(2)}" fill="#0f8f63" font-size="11" font-weight="700">Leading</text>
    <text x="${(x100 + 8).toFixed(2)}" y="${(plotTop + plotH - 8).toFixed(2)}" fill="#c76f24" font-size="11" font-weight="700">Weakening</text>
    <text x="${(plotLeft + 8).toFixed(2)}" y="${(plotTop + plotH - 8).toFixed(2)}" fill="#596878" font-size="11" font-weight="700">Lagging</text>
    <text x="${(plotLeft + 8).toFixed(2)}" y="${(plotTop + 16).toFixed(2)}" fill="#2679d2" font-size="11" font-weight="700">Improving</text>
  `;

  const counts = payload.quadrant_counts ?? {};
  const guide = payload.quadrant_guide ?? {};
  rrgLegend.innerHTML = `
    <span class="rrg-pill rrg-leading">Leading: ${counts.Leading ?? 0}</span>
    <span class="rrg-pill rrg-weakening">Weakening: ${counts.Weakening ?? 0}</span>
    <span class="rrg-pill rrg-lagging">Lagging: ${counts.Lagging ?? 0}</span>
    <span class="rrg-pill rrg-improving">Improving: ${counts.Improving ?? 0}</span>
    <div class="rrg-guide">
      <span class="rrg-guide-item"><strong>Leading:</strong> ${safeText(guide.Leading ?? "Strong relative strength with rising momentum.")}</span>
      <span class="rrg-guide-item"><strong>Weakening:</strong> ${safeText(guide.Weakening ?? "Still strong, but momentum is fading.")}</span>
      <span class="rrg-guide-item"><strong>Lagging:</strong> ${safeText(guide.Lagging ?? "Weak relative strength and weak momentum.")}</span>
      <span class="rrg-guide-item"><strong>Improving:</strong> ${safeText(guide.Improving ?? "Weak base, but momentum is improving.")}</span>
    </div>
  `;

  const val = payload.signal_validation ?? {};
  const leadHit = val.leading_hit_rate === null || val.leading_hit_rate === undefined ? "n/a" : `${(Number(val.leading_hit_rate) * 100).toFixed(1)}%`;
  const impHit = val.improving_hit_rate === null || val.improving_hit_rate === undefined ? "n/a" : `${(Number(val.improving_hit_rate) * 100).toFixed(1)}%`;
  const leadEx = val.leading_avg_excess === null || val.leading_avg_excess === undefined ? "n/a" : Number(val.leading_avg_excess).toFixed(2);
  const impEx = val.improving_avg_excess === null || val.improving_avg_excess === undefined ? "n/a" : Number(val.improving_avg_excess).toFixed(2);

  const trailDays = Number(payload.trail_days ?? state.rrgTrail ?? 7);
  const availableHistory = Number(payload.history_points_available ?? 0);
  const trailMode = state.rrgShowTrails ? `last ${trailDays} day(s)` : "off";
  const trailInfo = `Trail: ${trailMode}. `;
  const limitedTrailNote = availableHistory > 0 && availableHistory < trailDays
    ? `Only ${availableHistory} archived day(s) currently available. `
    : "";
  const historyNote = payload.history_note ? `${payload.history_note} ` : "";
  const axisNote = yOverrideNote ? `${yOverrideNote} ` : "";
  rrgValidation.textContent = `${trailInfo}${limitedTrailNote}${historyNote}${axisNote}Out-of-sample lens (${val.horizon_days ?? 7}d): Leading entries hit ${leadHit} (n=${val.leading_entries ?? 0}, excess ${leadEx}); Improving entries hit ${impHit} (n=${val.improving_entries ?? 0}, excess ${impEx}).`;
}

function exportRrgMomentumSheetAsPng() {
  const boards = Array.isArray(state.rrgMomentumLeaders) ? state.rrgMomentumLeaders : [];
  if (!boards.length) {
    setError("Momentum sheet export unavailable until leader lists are loaded.");
    return;
  }

  if (rrgMomentumExportPngBtn) {
    rrgMomentumExportPngBtn.disabled = true;
  }

  const pageWidth = 2550;
  const pageHeight = 3300;
  const margin = 120;
  const contentWidth = pageWidth - margin * 2;
  const panelGap = 36;

  const canvas = document.createElement("canvas");
  canvas.width = pageWidth;
  canvas.height = pageHeight;
  const ctx = canvas.getContext("2d");

  if (!ctx) {
    setError("Momentum sheet export failed: could not create canvas context.");
    if (rrgMomentumExportPngBtn) {
      rrgMomentumExportPngBtn.disabled = false;
    }
    return;
  }

  const g = ctx.createLinearGradient(0, 0, 0, pageHeight);
  g.addColorStop(0, "#f7fbff");
  g.addColorStop(1, "#eef6ff");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, pageWidth, pageHeight);

  const now = new Date();
  const generated = now.toLocaleString();
  const lookback = Number(state?.rrgMomentumMeta?.lookback ?? state.rrgLookback ?? 120);
  const trail = Number(state?.rrgMomentumMeta?.trail ?? state.rrgTrail ?? 7);

  let y = margin;

  ctx.fillStyle = "#183a56";
  ctx.font = "700 66px 'Segoe UI', Arial, sans-serif";
  ctx.fillText("MLB RS-Momentum Leaders Sheet", margin, y);
  y += 72;

  ctx.fillStyle = "#355972";
  ctx.font = "500 30px 'Segoe UI', Arial, sans-serif";
  ctx.fillText(`Generated ${generated} | Lookback ${lookback}d | Trail ${trail}d`, margin, y);
  y += 54;

  ctx.strokeStyle = "#c8dbee";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(margin, y);
  ctx.lineTo(pageWidth - margin, y);
  ctx.stroke();
  y += 34;

  ctx.fillStyle = "#244761";
  ctx.font = "700 38px 'Segoe UI', Arial, sans-serif";
  ctx.fillText("What Each Score Represents", margin, y);
  y += 44;

  const explanations = [
    {
      title: "Power Score",
      text: "Context-adjusted team strength score. Starts from Mscore and applies talent, form, risk, and context overlays (including leverage/clutch and coaching when signal is meaningful).",
    },
    {
      title: "D+ Score",
      text: "Balanced base/offense/defense/context model with extra emphasis on pitching quality and pitching health. Useful for baseline team quality and run-prevention consistency.",
    },
    {
      title: "Joe Score",
      text: "Fixed 45/40/15 blend: xwOBAcon quality (45%), inverse xFIP quality (40%), and clutch execution signal (15%), with reliability shrinkage and uncertainty control.",
    },
    {
      title: "RS-Momentum",
      text: "Relative momentum lens from the RRG engine. Values above 100 indicate improving relative momentum; below 100 indicate regressing relative momentum.",
    },
  ];

  ctx.fillStyle = "#2f516a";
  for (const row of explanations) {
    ctx.font = "700 28px 'Segoe UI', Arial, sans-serif";
    ctx.fillText(`${row.title}:`, margin, y);
    y += 34;
    ctx.font = "500 25px 'Segoe UI', Arial, sans-serif";
    y = drawWrappedCanvasText(ctx, row.text, margin + 10, y, contentWidth - 20, 34, 3);
    y += 12;
  }

  y += 18;

  const cardWidth = Math.floor((contentWidth - panelGap * 2) / 3);
  const cardHeight = 1450;
  const listStartY = 220;
  const lineStep = 48;

  const scoreOrder = ["live_power_score", "d_plus_score", "joe_score"];
  const scoreRank = Object.fromEntries(scoreOrder.map((k, i) => [k, i]));
  const orderedBoards = [...boards].sort((a, b) => {
    const ar = scoreRank[a.metricKey] ?? 99;
    const br = scoreRank[b.metricKey] ?? 99;
    return ar - br;
  });

  for (let i = 0; i < orderedBoards.length; i += 1) {
    const board = orderedBoards[i];
    const x = margin + i * (cardWidth + panelGap);
    const cardY = y;

    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#ccddef";
    ctx.lineWidth = 2;
    ctx.fillRect(x, cardY, cardWidth, cardHeight);
    ctx.strokeRect(x, cardY, cardWidth, cardHeight);

    ctx.fillStyle = "#1f4461";
    ctx.font = "700 34px 'Segoe UI', Arial, sans-serif";
    ctx.fillText(board.title || "Score", x + 24, cardY + 52);

    ctx.fillStyle = "#d9e8f5";
    ctx.fillRect(x + 20, cardY + 76, cardWidth - 40, 2);

    let iy = cardY + listStartY;
    ctx.fillStyle = "#35617b";
    ctx.font = "700 24px 'Segoe UI', Arial, sans-serif";
    ctx.fillText("Most Improving", x + 24, iy - 28);

    ctx.font = "600 23px 'Segoe UI', Arial, sans-serif";
    for (let rank = 0; rank < 5; rank += 1) {
      const row = board.improving?.[rank];
      const yy = iy + rank * lineStep;
      if (!row) {
        ctx.fillStyle = "#8aa0b3";
        ctx.fillText(`${rank + 1}. -`, x + 24, yy);
        continue;
      }
      ctx.fillStyle = "#2a4a63";
      const name = truncateCanvasText(ctx, `${rank + 1}. ${row.team}`, cardWidth - 220);
      ctx.fillText(name, x + 24, yy);
      const delta = Number(row.delta ?? (Number(row.momentum) - 100));
      const deltaText = Number.isFinite(delta) ? (delta >= 0 ? `+${delta.toFixed(2)}` : delta.toFixed(2)) : "0.00";
      ctx.fillStyle = "#0f8f63";
      ctx.textAlign = "right";
      ctx.fillText(`${Number(row.momentum).toFixed(2)} (${deltaText})`, x + cardWidth - 24, yy);
      ctx.textAlign = "left";
    }

    iy = cardY + listStartY + 370;
    ctx.fillStyle = "#35617b";
    ctx.font = "700 24px 'Segoe UI', Arial, sans-serif";
    ctx.fillText("Most Regressing", x + 24, iy - 28);

    ctx.font = "600 23px 'Segoe UI', Arial, sans-serif";
    for (let rank = 0; rank < 5; rank += 1) {
      const row = board.regressing?.[rank];
      const yy = iy + rank * lineStep;
      if (!row) {
        ctx.fillStyle = "#8aa0b3";
        ctx.fillText(`${rank + 1}. -`, x + 24, yy);
        continue;
      }
      ctx.fillStyle = "#2a4a63";
      const name = truncateCanvasText(ctx, `${rank + 1}. ${row.team}`, cardWidth - 220);
      ctx.fillText(name, x + 24, yy);
      const delta = Number(row.delta ?? (Number(row.momentum) - 100));
      const deltaText = Number.isFinite(delta) ? (delta >= 0 ? `+${delta.toFixed(2)}` : delta.toFixed(2)) : "0.00";
      ctx.fillStyle = "#a1482e";
      ctx.textAlign = "right";
      ctx.fillText(`${Number(row.momentum).toFixed(2)} (${deltaText})`, x + cardWidth - 24, yy);
      ctx.textAlign = "left";
    }
  }

  ctx.fillStyle = "#5a7389";
  ctx.font = "500 24px 'Segoe UI', Arial, sans-serif";
  ctx.fillText("Note: Momentum values are relative (centered around 100) and derived from the same RRG methodology used in-app.", margin, pageHeight - margin + 8);

  canvas.toBlob((blob) => {
    if (!blob) {
      setError("Momentum sheet export failed.");
      if (rrgMomentumExportPngBtn) {
        rrgMomentumExportPngBtn.disabled = false;
      }
      return;
    }

    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    const stamp = compactDateStamp(now);
    link.href = url;
    link.download = `momentum-leaders-sheet-${stamp}.png`;
    link.click();
    URL.revokeObjectURL(url);
    if (rrgMomentumExportPngBtn) {
      rrgMomentumExportPngBtn.disabled = false;
    }
  }, "image/png");
}

function exportAdvancedScatterAsPng() {
  if (!advancedScatterSvg || !advancedScatterSvg.innerHTML.trim()) {
    setError("Advanced chart export unavailable until the xFIP vs xwOBAcon chart is loaded.");
    return;
  }

  const serializer = new XMLSerializer();
  const svgClone = advancedScatterSvg.cloneNode(true);
  svgClone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  svgClone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");

  const viewBox = String(svgClone.getAttribute("viewBox") || "0 0 960 460")
    .trim()
    .split(/\s+/)
    .map((item) => Number(item));
  const exportWidth = Number.isFinite(viewBox[2]) ? Number(viewBox[2]) : 960;
  const exportHeight = Number.isFinite(viewBox[3]) ? Number(viewBox[3]) : 460;

  const svgBlob = new Blob([serializer.serializeToString(svgClone)], { type: "image/svg+xml;charset=utf-8" });
  const svgUrl = URL.createObjectURL(svgBlob);
  const image = new Image();

  image.onload = () => {
    const scale = 2;
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(exportWidth * scale));
    canvas.height = Math.max(1, Math.round(exportHeight * scale));

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      URL.revokeObjectURL(svgUrl);
      setError("Advanced chart export failed: could not create canvas context.");
      return;
    }

    ctx.fillStyle = "#f9fcff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.drawImage(image, 0, 0, exportWidth, exportHeight);
    URL.revokeObjectURL(svgUrl);

    canvas.toBlob((blob) => {
      if (!blob) {
        setError("Advanced chart export failed: PNG generation returned no data.");
        return;
      }
      const timestamp = new Date().toISOString().slice(0, 10);
      const teamPart = sanitizeFilePart(state.selectedTeam || "league");
      const downloadUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = `advanced-xfip-xwobacon-${teamPart}-${timestamp}.png`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(downloadUrl), 0);
      setError("");
    }, "image/png");
  };

  image.onerror = () => {
    URL.revokeObjectURL(svgUrl);
    setError("Advanced chart export failed.");
  };

  image.src = svgUrl;
}

function exportRrgAsPng() {
  if (!rrgSvg || !rrgSvg.innerHTML.trim()) {
    if (rrgValidation) {
      rrgValidation.textContent = "RRG export unavailable until chart data is loaded.";
    }
    return;
  }

  const serializer = new XMLSerializer();
  const svgClone = rrgSvg.cloneNode(true);
  svgClone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  svgClone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");

  const viewBox = String(svgClone.getAttribute("viewBox") || "0 0 960 460")
    .trim()
    .split(/\s+/)
    .map((item) => Number(item));
  const exportWidth = Number.isFinite(viewBox[2]) ? Number(viewBox[2]) : 960;
  const exportHeight = Number.isFinite(viewBox[3]) ? Number(viewBox[3]) : 460;

  const svgBlob = new Blob([serializer.serializeToString(svgClone)], { type: "image/svg+xml;charset=utf-8" });
  const svgUrl = URL.createObjectURL(svgBlob);
  const image = new Image();

  image.onload = () => {
    const scale = 2;
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(exportWidth * scale));
    canvas.height = Math.max(1, Math.round(exportHeight * scale));

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      URL.revokeObjectURL(svgUrl);
      return;
    }

    ctx.fillStyle = "#f9fcff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.drawImage(image, 0, 0, exportWidth, exportHeight);
    URL.revokeObjectURL(svgUrl);

    canvas.toBlob((blob) => {
      if (!blob) {
        return;
      }
      const timestamp = new Date().toISOString().slice(0, 10);
      const metric = state.rrgMetric || "power";
      const downloadUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = `rrg-${metric}-${timestamp}.png`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(downloadUrl), 0);
    }, "image/png");
  };

  image.onerror = () => {
    URL.revokeObjectURL(svgUrl);
    if (rrgValidation) {
      rrgValidation.textContent = "RRG export failed.";
    }
  };

  image.src = svgUrl;
}

async function loadRrgData() {
  if (!rrgSvg) {
    return;
  }

  updateRrgLabels();

  const metric = state.rrgMetric || "power";
  const lookback = Math.max(30, Math.min(365, Number(state.rrgLookback || 120)));
  const trail = Math.max(3, Math.min(45, Number(state.rrgTrail || 7)));
  const minPoints = Math.max(4, Math.min(60, Math.floor(trail / 2) + 2));

  const params = new URLSearchParams({
    metric,
    lookback_days: String(lookback),
    trail_days: String(trail),
    min_points: String(minPoints),
  });

  try {
    const response = await fetch(`/api/rrg?${params.toString()}`);
    if (!response.ok) {
      throw new Error(`RRG request failed (${response.status})`);
    }

    const payload = await response.json();
    state.rrgPayload = payload;
    syncRrgTeamSelection(payload);
    renderRrgTeamSelector();
    renderRrg(payload);
    await loadRrgMomentumLeaders(payload, metric);
  } catch (error) {
    renderRrgMomentumLeaders([], { message: "RRG unavailable." });
    rrgSvg.innerHTML = `<text x="24" y="42" fill="#a63d2b" font-size="15">Could not load RRG data: ${safeText(error.message)}</text>`;
    if (rrgValidation) {
      rrgValidation.textContent = "RRG unavailable.";
    }
  }
}

function renderSummary(snapshot) {
  const summary = snapshot.summary;
  const cards = [
    { label: "Top Team", value: summary.top_team ?? "-" },
    { label: "Top Live Power", value: formatNumber(summary.top_live_power_score, 2) },
    { label: "Avg Mscore", value: formatNumber(summary.average_mscore, 3) },
    { label: "Avg Win%", value: formatPercent(summary.average_win_pct, 1) },
    { label: "Games Today", value: formatNumber(summary.total_games_today) },
    { label: "Completed", value: formatNumber(summary.completed_games_today) },
    { label: "Live / Upcoming", value: formatNumber(summary.live_games_today) },
    { label: "Market Matchups", value: formatNumber(summary.market_matchups_today) },
    { label: "Actionable Value", value: formatNumber(summary.actionable_value_spots) },
    { label: "High Quality (A)", value: formatNumber(summary.high_quality_value_spots) },
  ];

  summaryCards.innerHTML = cards
    .map(
      (card) => `
      <article class="stat-card">
        <p class="stat-label">${safeText(card.label)}</p>
        <p class="stat-value">${safeText(card.value)}</p>
      </article>
    `,
    )
    .join("");
}

function renderTeams(snapshot) {
  const filter = state.filterText.trim().toLowerCase();
  const rows = (snapshot.teams ?? []).filter((team) => team.team.toLowerCase().includes(filter));

  teamsBody.innerHTML = rows
    .map((team) => {
      const delta = signedDelta(team.rank_delta);
      const selectedClass = team.team === state.selectedTeam ? "selected-team-row" : "";
      const gapRaw = Number(team.power_minus_d_plus ?? team.power_minus_mscore);
      const hasGap = Number.isFinite(gapRaw);
      const gapText = !hasGap ? "-" : (gapRaw > 0 ? `+${gapRaw.toFixed(3)}` : gapRaw.toFixed(3));
      const gapClass = !hasGap ? "delta-flat" : (gapRaw > 0 ? "delta-up" : (gapRaw < 0 ? "delta-down" : "delta-flat"));

      return `
      <tr class="team-row ${selectedClass}" data-team="${safeText(team.team)}">
        <td>${safeText(team.live_rank)}</td>
        <td>${safeText(team.workbook_rank ?? "-")}</td>
        <td>${safeText(team.team)}</td>
        <td>${safeText(formatNumber(team.mscore, 3))}</td>
        <td>${safeText(formatNumber(team.live_power_score, 2))}</td>
        <td>${safeText(formatNumber(team.d_plus_score, 2))}</td>
        <td>${safeText(formatNumber(team.joe_score, 2))}</td>
        <td><span class="${gapClass}">${safeText(gapText)}</span></td>
        <td>${safeText(team.wins)}-${safeText(team.losses)}</td>
        <td>${safeText(formatPercent(team.win_pct, 1))}</td>
        <td>${safeText(team.run_diff)}</td>
        <td>${safeText(team.streak)}</td>
        <td>${safeText(formatNumber(team.projected_wins_162, 1))} <span class="${delta.cls}">(${delta.text})</span></td>
      </tr>
    `;
    })
    .join("");
}

function renderRankingHeaderHints(snapshot) {
  const entries = snapshot.stat_catalog?.ranking_table ?? [];
  const hints = Object.fromEntries(entries.map((entry) => [entry.key, entry.description]));

  document.querySelectorAll("thead th[data-column]").forEach((header) => {
    const key = header.getAttribute("data-column");
    const hint = hints[key] ?? "Ranking metric in the dashboard table.";
    header.setAttribute("title", hint);

    if (key !== "team") {
      header.setAttribute("data-stat-id", statId("ranking", key));
    } else {
      header.removeAttribute("data-stat-id");
    }
  });
}

function gameStatusClass(stateValue) {
  if (stateValue === "Final") return "status-pill status-final";
  if (stateValue === "Live" || stateValue === "In Progress") return "status-pill status-live";
  return "status-pill";
}

function formatOfficialDate(value) {
  if (!value) {
    return "";
  }
  const date = new Date(`${value}T12:00:00`);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function renderGames(snapshot) {
  const games = snapshot.games_today ?? [];
  if (!games.length) {
    gamesList.innerHTML = "<p>No live or recent games in the current window.</p>";
    return;
  }

  gamesList.innerHTML = games
    .map(
      (game) => `
      <div class="game-row">
        <div class="game-line">
          <span>${safeText(game.away_team)}</span>
          <strong>${safeText(game.away_score ?? "-")}</strong>
        </div>
        <div class="game-line">
          <span>${safeText(game.home_team)}</span>
          <strong>${safeText(game.home_score ?? "-")}</strong>
        </div>
        <div class="game-line">
          <span>${safeText(formatOfficialDate(game.official_date))}</span>
          <span>${safeText(game.venue ?? "")}</span>
        </div>
        <span class="${gameStatusClass(game.state)}">${safeText(game.status)}${game.inning ? ` - ${safeText(game.inning)}` : ""}</span>
      </div>
    `,
    )
    .join("");
}

function renderPredictions(snapshot) {
  const games = snapshot.matchup_predictions ?? [];
  const model = snapshot.prediction_engine ?? {};

  if (!games.length) {
    predictionBars.innerHTML = "<p class='muted-note'>No matchup predictions available yet.</p>";
    return;
  }

  const metrics = model.metrics_test ?? {};
  const cv = model.rolling_cv_mean ?? {};
  const cal = model.calibration_test ?? {};
  const metricParts = [];

  if (metrics.accuracy !== undefined) {
    metricParts.push(`Test Acc ${formatPercent(metrics.accuracy, 1)}`);
  }
  if (metrics.log_loss !== undefined) {
    metricParts.push(`Test LogLoss ${formatNumber(metrics.log_loss, 3)}`);
  }
  if (cv.accuracy !== undefined) {
    metricParts.push(`CV Acc ${formatPercent(cv.accuracy, 1)}`);
  }
  if (model.blend_weight !== undefined && model.blend_weight !== null) {
    metricParts.push(`Blend ${formatNumber(model.blend_weight, 2)}`);
  }
  if (cal.ece !== undefined && cal.ece !== null) {
    metricParts.push(`ECE ${(Number(cal.ece) * 100).toFixed(1)}%`);
  }
  const homeBaseRate = Number(model.home_prior_base_rate);
  const homeNeutralizeStrength = Number(model.home_prior_neutralize_strength ?? 0);
  if (Number.isFinite(homeBaseRate) && homeNeutralizeStrength > 0) {
    metricParts.push(`Home Prior ${formatPercent(homeBaseRate, 1)} x ${formatNumber(homeNeutralizeStrength, 2)}`);
  }

  const metricLine = metricParts.length
    ? `<p class="muted-note">Backtest ${safeText(model.season ?? "")} (${safeText(model.model_variant ?? "v4")}): ${safeText(metricParts.join(" | "))}</p>`
    : "";

  const rows = games
    .map((game) => {
      const favoredProb = Number(game.favored_win_prob ?? 0.5);
      const modelHome = Number(game.model_home_win_prob ?? game.home_win_prob ?? 0.5);
      const rawCandidate = Number(game.model_home_win_prob_raw);
      const modelHomeRaw = Number.isFinite(rawCandidate) ? rawCandidate : null;
      const explicitHomePriorAdj = Number(game.home_prior_neutralization);
      const homePriorAdj = Number.isFinite(explicitHomePriorAdj)
        ? explicitHomePriorAdj
        : (modelHomeRaw === null ? 0 : (modelHome - modelHomeRaw));
      const modelText = modelHomeRaw === null
        ? `Model H ${formatPercent(modelHome, 1)}`
        : `Model H ${formatPercent(modelHome, 1)} (raw ${formatPercent(modelHomeRaw, 1)} | prior ${formatSignedPctPoints(homePriorAdj)})`;
      const starterAdj = Number(game.starter_adjustment ?? 0);
      const splitAdj = Number(game.split_adjustment ?? 0);
      const bullpenAdj = Number(game.bullpen_adjustment ?? 0);
      const travelAdj = Number(game.travel_adjustment ?? 0);
      const lineupAdj = Number(game.lineup_adjustment ?? 0);
      const bullpenHealthAdj = Number(game.bullpen_health_adjustment ?? 0);
      const lineupHealthAdj = Number(game.lineup_health_adjustment ?? 0);
      const luckAdj = Number(game.luck_adjustment ?? 0);
      const advancedAdj = Number(game.advanced_adjustment ?? 0);
      const totalAdj = starterAdj + splitAdj + bullpenAdj + travelAdj + lineupAdj + bullpenHealthAdj + lineupHealthAdj + luckAdj + advancedAdj;
      const marketProb = game.market_home_win_prob;
      const adjText = `Adj ${(100 * totalAdj).toFixed(1)}% (SP ${(100 * starterAdj).toFixed(1)} | Pen ${(100 * (bullpenAdj + bullpenHealthAdj)).toFixed(1)} | Line ${(100 * (lineupAdj + lineupHealthAdj)).toFixed(1)} | Luck ${(100 * luckAdj).toFixed(1)} | Adv ${(100 * advancedAdj).toFixed(1)})`;
      const marketText = marketProb === null || marketProb === undefined ? "" : ` | Market H ${formatPercent(marketProb, 1)}`;
      const bandLow = game.home_win_prob_band_low === null || game.home_win_prob_band_low === undefined ? null : Number(game.home_win_prob_band_low);
      const bandHigh = game.home_win_prob_band_high === null || game.home_win_prob_band_high === undefined ? null : Number(game.home_win_prob_band_high);
      const bandText = (bandLow === null || bandHigh === null) ? "Band N/A" : `Band ${formatPercent(bandLow, 1)}-${formatPercent(bandHigh, 1)}`;
      const uncLevel = String(game.uncertainty_level ?? "low").toLowerCase();
      const uncCls = uncLevel === "high" ? "uncertainty-tag high" : (uncLevel === "medium" ? "uncertainty-tag medium" : "uncertainty-tag low");
      const uncLabel = uncLevel === "high" ? "High Unc" : (uncLevel === "medium" ? "Med Unc" : "Low Unc");

      const matchupId = getMatchupId(game);
      const starterOpen = Boolean(state.starterCompareOpen?.[matchupId]);
      const starterBtnLabel = starterOpen ? "Hide SP" : "SP Compare";

      return `
      <div class="bar-row is-clickable ${uncLevel === "high" ? "bar-row-high-unc" : ""}" data-matchup-id="${safeText(matchupId)}" role="button" tabindex="0" title="Open matchup breakdown">
        <div class="bar-meta">
          <span>${safeText(game.away_team)} @ ${safeText(game.home_team)}</span>
          <span>${safeText(game.favored_team)} ${safeText(formatPercent(favoredProb, 1))}</span>
        </div>
        <div class="bar-meta">
          <span class="muted-note">${safeText(modelText)} | ${safeText(adjText)}${safeText(marketText)}</span>
          <span class="muted-note">${safeText(game.away_probable_pitcher ?? "TBD")} vs ${safeText(game.home_probable_pitcher ?? "TBD")}</span>
        </div>
        <div class="bar-meta bar-meta-actions">
          <span class="muted-note">${safeText(bandText)}${game.uncertainty_note ? ` | ${safeText(game.uncertainty_note)}` : ""}</span>
          <span class="bar-row-actions">
            <span class="${uncCls}">${safeText(uncLabel)}</span>
            <button type="button" class="sp-compare-btn" data-sp-compare="1" data-matchup-id="${safeText(matchupId)}">${safeText(starterBtnLabel)}</button>
          </span>
        </div>
        ${renderStarterComparisonPanel(game, matchupId, starterOpen)}
        <div class="bar-track">
          <div class="bar-fill" style="width: ${Math.max(2, favoredProb * 100)}%"></div>
        </div>
      </div>
      `;
    })
    .join("");

  const slateLine = `<p class="muted-note">Today: ${safeText(String(games.length))} game${games.length === 1 ? "" : "s"}</p>`;
  predictionBars.innerHTML = `${metricLine}${slateLine}${rows}`;
}

function renderValueFinder(snapshot) {
  if (!valueFinderList) {
    return;
  }

  const rows = buildValueFinderRows(snapshot);
  if (!rows.length) {
    valueFinderList.innerHTML = "<p class='muted-note'>No value rows yet.</p>";
    return;
  }

  const marketEnabled = Boolean(snapshot?.prediction_engine?.market_odds_enabled);
  const marketStatus = String(snapshot?.prediction_engine?.market_status ?? "");
  const marketRows = rows.filter((row) => row.market_available);

  let marketPart = `Market rows ${marketRows.length}/${rows.length}`;
  if (marketStatus === "disabled_missing_api_key") {
    marketPart += " (missing MSCORE_ODDS_API_KEY)";
  } else if (marketStatus === "enabled_no_lines_returned") {
    marketPart += " (no lines returned yet)";
  } else if (!marketEnabled) {
    marketPart += " (feed unavailable)";
  }

  const highUncCount = rows.filter((row) => String(row.uncertainty_level ?? "").toLowerCase() === "high").length;
  const actionableCount = rows.filter((row) => Boolean(row.bet_quality_actionable)).length;
  const highQualityCount = rows.filter((row) => Number(row.bet_quality_score ?? 0) >= 75).length;
  const summary = `<p class="muted-note">${safeText(marketPart)} | Actionable ${safeText(String(actionableCount))} | High quality ${safeText(String(highQualityCount))} | Upset calls ${rows.filter((row) => row.is_market_upset).length} | High uncertainty ${safeText(String(highUncCount))}</p>`;
  const setupHint = marketStatus === "disabled_missing_api_key"
    ? `<p class="muted-note">Set <code>MSCORE_ODDS_API_KEY</code> before starting the server to enable true market edge and upset detection.</p>`
    : "";

  const cards = rows
    .map((row) => {
      const edge = row.market_edge_pick;
      const edgeRaw = row.market_edge_pick_raw === null || row.market_edge_pick_raw === undefined ? null : Number(row.market_edge_pick_raw);
      const edgeText = edge === null || edge === undefined ? "No market line" : formatSignedPctPoints(edge);
      const edgeRawText = edgeRaw === null || edgeRaw === undefined ? null : formatSignedPctPoints(edgeRaw);
      const edgeClass = edge === null || edge === undefined
        ? "delta-flat"
        : (Number(edge) > 0 ? "delta-up" : (Number(edge) < 0 ? "delta-down" : "delta-flat"));
      const tier = row.value_tier ?? computeValueTier(edge);
      const tierLabel = formatValueTier(tier);
      const tierCls = `tier-${safeText(String(tier).toLowerCase())}`;

      const qualityScore = Number(row.bet_quality_score ?? NaN);
      const qualityGrade = formatBetQualityGrade(row.bet_quality_grade);
      const qualityLabel = Number.isFinite(qualityScore) ? `${qualityScore.toFixed(1)} ${qualityGrade}` : qualityGrade;
      const qualityCls = `quality-${safeText(String(qualityGrade).toLowerCase())}`;
      const qualityTag = `<span class="value-tag ${qualityCls}">Q ${safeText(qualityLabel)}</span>`;
      const actionableTag = row.bet_quality_actionable ? `<span class="value-tag quality-action">Actionable</span>` : "";

      const upsetTag = row.is_market_upset ? `<span class="value-tag upset">Upset</span>` : "";
      const uncLevel = String(row.uncertainty_level ?? "low").toLowerCase();
      const uncTag = uncLevel === "high"
        ? `<span class="value-tag uncertainty-high">High Unc</span>`
        : (uncLevel === "medium" ? `<span class="value-tag uncertainty-med">Med Unc</span>` : "");

      const modelOddsText = Number.isFinite(Number(row.model_pick_prob)) ? formatPercent(row.model_pick_prob, 1) : "-";
      const marketOddsText = row.market_pick_prob === null || row.market_pick_prob === undefined ? "N/A" : formatPercent(row.market_pick_prob, 1);
      const finalOddsText = row.final_pick_prob === null || row.final_pick_prob === undefined ? "N/A" : formatPercent(row.final_pick_prob, 1);
      const edgeMeta = edgeRawText && edgeRawText !== edgeText ? ` | Raw ${safeText(edgeRawText)}` : "";

      const deltaFinal = numberOrNull(row.delta_home_win_prob);
      const deltaPre = numberOrNull(row.delta_pre_market_home_win_prob);
      const deltaDate = row.delta_source_date ? String(row.delta_source_date) : null;
      const deltaParts = [];
      if (Number.isFinite(deltaFinal)) {
        deltaParts.push(`Final ${formatSignedPctPoints(deltaFinal)}`);
      }
      if (Number.isFinite(deltaPre)) {
        deltaParts.push(`Pre ${formatSignedPctPoints(deltaPre)}`);
      }
      const deltaLine = deltaParts.length
        ? `<p class="value-row-odds muted-note">Delta${deltaDate ? ` vs ${safeText(deltaDate)}` : ""}: ${safeText(deltaParts.join(" | "))}</p>`
        : "";

      const lowQualityClass = !row.bet_quality_actionable && Number.isFinite(qualityScore) && qualityScore < 45 ? "value-row-pass" : "";

      return `
      <article class="value-row is-clickable ${uncLevel === "high" ? "value-row-high-unc" : ""} ${lowQualityClass}" data-matchup-id="${safeText(row.matchup_id)}" role="button" tabindex="0" title="Open matchup breakdown">
        <div class="value-row-top">
          <span>${safeText(row.away_team)} @ ${safeText(row.home_team)}</span>
          <span class="value-tag ${tierCls}">${safeText(tierLabel)}</span>
        </div>
        <div class="value-row-mid">
          <span>${safeText(row.model_pick_team ?? "Model Pick")} | Edge</span>
          <span class="${edgeClass}">${safeText(edgeText)}</span>
        </div>
        <p class="value-row-odds muted-note">Model (pre): ${safeText(modelOddsText)} | Market: ${safeText(marketOddsText)} | Final: ${safeText(finalOddsText)}${edgeMeta}</p>
        ${deltaLine}
        <div class="value-row-bot">
          <span class="muted-note">Conf ${safeText(formatPercent(row.confidence ?? 0, 1))}${row.uncertainty_note ? ` | ${safeText(row.uncertainty_note)}` : ""}</span>
          <span class="value-tag-group">${qualityTag}${actionableTag}${uncTag}${upsetTag}</span>
        </div>
      </article>
      `;
    })
    .join("");

  valueFinderList.innerHTML = `${summary}${setupHint}${cards}`;
}

function renderTopChart(snapshot) {
  const top = (snapshot.teams ?? []).slice(0, 10);
  topTeamsChart.innerHTML = top
    .map(
      (team) => `
      <div class="bar-row">
        <div class="bar-meta">
          <span>${safeText(team.live_rank)}. ${safeText(team.team)}</span>
          <span>${safeText(formatNumber(team.live_power_score, 1))}</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width: ${Math.max(2, Number(team.live_power_score || 0))}%"></div>
        </div>
      </div>
    `,
    )
    .join("");
}

function renderAdvancedPanel(snapshot) {
  const hasAnyTarget = Boolean(advancedSummary || advancedScatterSvg || advancedTeamBars || advancedExpectedGrid || advancedTeamTitle);
  if (!hasAnyTarget) {
    return;
  }

  const setSummary = (html) => {
    if (advancedSummary) {
      advancedSummary.innerHTML = html;
    }
  };
  const setScatter = (html) => {
    if (advancedScatterSvg) {
      advancedScatterSvg.innerHTML = html;
    }
  };
  const setBars = (html) => {
    if (advancedTeamBars) {
      advancedTeamBars.innerHTML = html;
    }
  };
  const setExpected = (html) => {
    if (advancedExpectedGrid) {
      advancedExpectedGrid.innerHTML = html;
    }
  };

  try {
    const teams = snapshot?.teams ?? [];
    if (!teams.length) {
      setSummary("<p class='muted-note'>No advanced team data.</p>");
      setScatter("<text x='24' y='40' fill='#647b8f' font-size='15'>No advanced data.</text>");
      setBars("");
      setExpected("");
      return;
    }

    const rows = teams.map((team) => {
      const xw = Number(getTeamLiveValue(team, "xwobacon_proxy"));
      const xfip = Number(getTeamLiveValue(team, "xfip_proxy"));
      const ops = Number(getTeamMlbValue(team, "mlb::season::hitting::ops"));
      const era = Number(getTeamMlbValue(team, "mlb::season::pitching::era"));
      const kbb = toRatioValue(getTeamMlbValue(team, "mlb::seasonAdvanced::pitching::strikeoutsMinusWalksPercentage"));
      const hardHit = toRatioValue(getTeamMlbValue(team, "mlb::seasonAdvanced::hitting::hardHitPercentage"));
      const whiff = toRatioValue(getTeamMlbValue(team, "mlb::seasonAdvanced::pitching::whiffPercentage"));

      return {
        team: team.team,
        xwobacon: Number.isFinite(xw) ? xw : null,
        xfip: Number.isFinite(xfip) ? xfip : null,
        ops: Number.isFinite(ops) ? ops : null,
        era: Number.isFinite(era) ? era : null,
        kbb,
        hardHit,
        whiff,
      };
    });

    const scatterRows = rows.filter((row) => Number.isFinite(Number(row.xwobacon)) && Number.isFinite(Number(row.xfip)));
    if (!scatterRows.length) {
      setSummary("<p class='muted-note'>Advanced proxies not available yet.</p>");
      setScatter("<text x='24' y='40' fill='#647b8f' font-size='15'>xwOBAcon/xFIP proxies unavailable.</text>");
      setBars("<p class='muted-note'>No percentile bars available.</p>");
      setExpected("");
      return;
    }

    const xwValues = scatterRows.map((row) => Number(row.xwobacon));
    const xfipValues = scatterRows.map((row) => Number(row.xfip));
    const avgXw = average(xwValues);
    const avgXfip = average(xfipValues);
    const strongBoth = scatterRows.filter((row) => Number(row.xwobacon) >= Number(avgXw) && Number(row.xfip) <= Number(avgXfip)).length;
    const advGames = (snapshot.matchup_predictions ?? []).filter((row) => Math.abs(Number(row?.advanced_adjustment ?? 0)) >= 0.01).length;

    setSummary(
      [
        { label: "League Avg xwOBAcon", value: formatNumber(avgXw, 3) },
        { label: "League Avg xFIP", value: formatNumber(avgXfip, 3) },
        { label: "Strong Both (Teams)", value: String(strongBoth) },
        { label: "Today Adv-Edge Games", value: String(advGames) },
      ]
        .map(
          (row) => `
      <article class="advanced-pill">
        <p class="advanced-pill-label">${safeText(row.label)}</p>
        <p class="advanced-pill-value">${safeText(row.value)}</p>
      </article>
      `,
        )
        .join(""),
    );

    const selectedTeamName = state.selectedTeam || teams[0].team;
    const selectedRow = rows.find((row) => row.team === selectedTeamName) || rows[0];
    if (!selectedRow) {
      setBars("<p class='muted-note'>No percentile bars available.</p>");
      setExpected("");
      return;
    }
    if (advancedTeamTitle) {
      advancedTeamTitle.textContent = `${selectedRow.team} Percentile Lens`;
    }

    const xMinRaw = Math.min(...xwValues);
    const xMaxRaw = Math.max(...xwValues);
    const yMinRaw = Math.min(...xfipValues);
    const yMaxRaw = Math.max(...xfipValues);

    const xPad = Math.max(0.01, (xMaxRaw - xMinRaw) * 0.15);
    const yPad = Math.max(0.08, (yMaxRaw - yMinRaw) * 0.18);
    const xMin = xMinRaw - xPad;
    const xMax = xMaxRaw + xPad;
    const yMin = yMinRaw - yPad;
    const yMax = yMaxRaw + yPad;

    const width = 960;
    const height = 460;
    const pad = { left: 72, right: 24, top: 24, bottom: 58 };
    const plotX = pad.left;
    const plotY = pad.top;
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const sx = (value) => plotX + ((Number(value) - xMin) / Math.max(1e-9, xMax - xMin)) * plotW;
    const sy = (value) => plotY + ((yMax - Number(value)) / Math.max(1e-9, yMax - yMin)) * plotH;

    const xMid = sx(avgXw);
    const yMid = sy(avgXfip);

    const pointsMarkup = scatterRows
      .map((row) => {
        const px = sx(row.xwobacon);
        const py = sy(row.xfip);
        const selected = row.team === selectedRow.team;
        const color = selected ? "#0f8f63" : "#2f6fa8";
        const r = selected ? 8.5 : 5.2;
        const label = selected ? row.team : shortTeamName(row.team);
        return `
      <g>
        <circle cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="${r.toFixed(2)}" fill="${color}" fill-opacity="0.86" stroke="#ffffff" stroke-width="1.3" />
        <text x="${(px + (selected ? 10 : 7)).toFixed(2)}" y="${(py - (selected ? 9 : 7)).toFixed(2)}" fill="#2a4a66" font-size="${selected ? "12" : "10"}" font-weight="${selected ? "700" : "600"}">${safeText(label)}</text>
      </g>
      `;
      })
      .join("");

    const xTicks = Array.from({ length: 5 }).map((_, i) => xMin + (i * (xMax - xMin)) / 4);
    const yTicks = Array.from({ length: 5 }).map((_, i) => yMin + (i * (yMax - yMin)) / 4);

    setScatter(`
    <rect x="${plotX}" y="${plotY}" width="${plotW}" height="${plotH}" fill="#f9fcff" stroke="#d1e1ee" stroke-width="1" />
    <rect x="${xMid.toFixed(2)}" y="${plotY}" width="${(plotX + plotW - xMid).toFixed(2)}" height="${(yMid - plotY).toFixed(2)}" fill="rgba(15,143,99,0.10)" />
    <rect x="${xMid.toFixed(2)}" y="${yMid.toFixed(2)}" width="${(plotX + plotW - xMid).toFixed(2)}" height="${(plotY + plotH - yMid).toFixed(2)}" fill="rgba(255,154,74,0.10)" />
    <rect x="${plotX}" y="${yMid.toFixed(2)}" width="${(xMid - plotX).toFixed(2)}" height="${(plotY + plotH - yMid).toFixed(2)}" fill="rgba(120,132,145,0.11)" />
    <rect x="${plotX}" y="${plotY}" width="${(xMid - plotX).toFixed(2)}" height="${(yMid - plotY).toFixed(2)}" fill="rgba(38,121,210,0.10)" />

    ${xTicks
      .map((tick) => {
        const tx = sx(tick);
        return `<g><line x1="${tx.toFixed(2)}" y1="${plotY}" x2="${tx.toFixed(2)}" y2="${(plotY + plotH).toFixed(2)}" stroke="#e7eef5" stroke-width="1" /><text x="${(tx - 14).toFixed(2)}" y="${(plotY + plotH + 20).toFixed(2)}" fill="#5e7488" font-size="10">${tick.toFixed(3)}</text></g>`;
      })
      .join("")}

    ${yTicks
      .map((tick) => {
        const ty = sy(tick);
        return `<g><line x1="${plotX}" y1="${ty.toFixed(2)}" x2="${(plotX + plotW).toFixed(2)}" y2="${ty.toFixed(2)}" stroke="#e7eef5" stroke-width="1" /><text x="${(plotX - 44).toFixed(2)}" y="${(ty + 4).toFixed(2)}" fill="#5e7488" font-size="10">${tick.toFixed(2)}</text></g>`;
      })
      .join("")}

    <line x1="${plotX}" y1="${yMid.toFixed(2)}" x2="${(plotX + plotW).toFixed(2)}" y2="${yMid.toFixed(2)}" stroke="#2f4254" stroke-width="1.4" stroke-dasharray="6 5" />
    <line x1="${xMid.toFixed(2)}" y1="${plotY}" x2="${xMid.toFixed(2)}" y2="${(plotY + plotH).toFixed(2)}" stroke="#2f4254" stroke-width="1.4" stroke-dasharray="6 5" />

    ${pointsMarkup}

    <text x="${(plotX + plotW / 2 - 92).toFixed(2)}" y="${(height - 14).toFixed(2)}" fill="#2f4b63" font-size="12" font-weight="600">xwOBAcon Proxy (higher is better)</text>
    <text x="16" y="${(plotY + plotH / 2).toFixed(2)}" fill="#2f4b63" font-size="12" font-weight="600" transform="rotate(-90 16 ${(plotY + plotH / 2).toFixed(2)})">xFIP Proxy (lower is better)</text>
  `);

    const metrics = [
      {
        label: "xwOBAcon Proxy",
        value: selectedRow.xwobacon,
        values: rows.map((r) => r.xwobacon),
        higherBetter: true,
        format: (v) => formatNumber(v, 3),
      },
      {
        label: "xFIP Proxy",
        value: selectedRow.xfip,
        values: rows.map((r) => r.xfip),
        higherBetter: false,
        format: (v) => formatNumber(v, 3),
      },
      {
        label: "K-BB%",
        value: selectedRow.kbb,
        values: rows.map((r) => r.kbb),
        higherBetter: true,
        format: (v) => `${(Number(v) * 100).toFixed(1)}%`,
      },
      {
        label: "HardHit%",
        value: selectedRow.hardHit,
        values: rows.map((r) => r.hardHit),
        higherBetter: true,
        format: (v) => `${(Number(v) * 100).toFixed(1)}%`,
      },
      {
        label: "Whiff%",
        value: selectedRow.whiff,
        values: rows.map((r) => r.whiff),
        higherBetter: true,
        format: (v) => `${(Number(v) * 100).toFixed(1)}%`,
      },
    ];

    setBars(
      metrics
        .map((metric) => {
          const pct = percentileRank(metric.values, metric.value, metric.higherBetter);
          const valueText = metric.value === null || metric.value === undefined || Number.isNaN(Number(metric.value))
            ? "-"
            : metric.format(metric.value);
          const pctText = pct === null ? "-" : `${pct.toFixed(0)}th`;
          const width = pct === null ? 2 : Math.max(2, Math.min(100, pct));
          return `
      <div class="advanced-bar-row">
        <div class="advanced-bar-meta">
          <span>${safeText(metric.label)}</span>
          <strong>${safeText(valueText)} (${safeText(pctText)})</strong>
        </div>
        <div class="advanced-bar-track">
          <div class="advanced-bar-fill" style="width:${width.toFixed(1)}%"></div>
        </div>
      </div>
      `;
        })
        .join(""),
    );

    const xwPct = percentileRank(rows.map((r) => r.xwobacon), selectedRow.xwobacon, true);
    const opsPct = percentileRank(rows.map((r) => r.ops), selectedRow.ops, true);
    const xfipPct = percentileRank(rows.map((r) => r.xfip), selectedRow.xfip, false);
    const eraPct = percentileRank(rows.map((r) => r.era), selectedRow.era, false);
    const kbbPct = percentileRank(rows.map((r) => r.kbb), selectedRow.kbb, true);
    const whiffPct = percentileRank(rows.map((r) => r.whiff), selectedRow.whiff, true);

    const contactDelta = xwPct === null || opsPct === null ? null : xwPct - opsPct;
    const pitchDelta = xfipPct === null || eraPct === null ? null : xfipPct - eraPct;
    const shapeDelta = kbbPct === null || whiffPct === null ? null : kbbPct - whiffPct;

    const describeDelta = (delta) => {
      if (delta === null) {
        return "Insufficient data";
      }
      if (delta >= 6) {
        return "Underlying stronger than current results";
      }
      if (delta <= -6) {
        return "Current results ahead of underlying profile";
      }
      return "Underlying and current results are aligned";
    };

    const cards = [
      {
        title: "Contact vs Output",
        main: formatPctPoints(contactDelta),
        sub: `xwOBAcon pctile ${xwPct === null ? "-" : xwPct.toFixed(0)} vs OPS pctile ${opsPct === null ? "-" : opsPct.toFixed(0)}. ${describeDelta(contactDelta)}.`,
        cls: contactDelta === null ? "delta-flat" : (contactDelta >= 0 ? "delta-up" : "delta-down"),
      },
      {
        title: "Pitching Underlying vs ERA",
        main: formatPctPoints(pitchDelta),
        sub: `xFIP pctile ${xfipPct === null ? "-" : xfipPct.toFixed(0)} vs ERA pctile ${eraPct === null ? "-" : eraPct.toFixed(0)}. ${describeDelta(pitchDelta)}.`,
        cls: pitchDelta === null ? "delta-flat" : (pitchDelta >= 0 ? "delta-up" : "delta-down"),
      },
      {
        title: "K-BB vs Whiff Shape",
        main: formatPctPoints(shapeDelta),
        sub: `K-BB pctile ${kbbPct === null ? "-" : kbbPct.toFixed(0)} vs Whiff pctile ${whiffPct === null ? "-" : whiffPct.toFixed(0)}.`,
        cls: shapeDelta === null ? "delta-flat" : (shapeDelta >= 0 ? "delta-up" : "delta-down"),
      },
    ];

    setExpected(
      cards
        .map(
          (card) => `
      <article class="advanced-expected-card">
        <p class="advanced-expected-title">${safeText(card.title)}</p>
        <p class="advanced-expected-main ${safeText(card.cls)}">${safeText(card.main)}</p>
        <p class="advanced-expected-sub">${safeText(card.sub)}</p>
      </article>
      `,
        )
        .join(""),
    );
  } catch (error) {
    console.error("Advanced Stats lens render failed", error);
    const message = safeText(error?.message ?? "Unknown rendering error");
    setSummary(`<p class='muted-note'>Advanced Stats failed to render: ${message}</p>`);
    setScatter("<text x='24' y='40' fill='#647b8f' font-size='15'>Render error. Check console.</text>");
    setBars("<p class='muted-note'>Render error in percentile lens.</p>");
    setExpected("");
  }
}


function renderTeamHighlights(team) {
  const delta = signedDelta(team.rank_delta);
  const highlights = [
    { label: "Live Rank", value: team.live_rank },
    { label: "Template Rank", value: team.workbook_rank ?? "-" },
    { label: "Record", value: `${team.wins}-${team.losses}` },
    { label: "Live Power", value: formatNumber(team.live_power_score, 2) },
    { label: "Projected Wins", value: formatNumber(team.projected_wins_162, 1) },
    { label: "Rank Delta", value: `${delta.text}` },
  ];

  teamHighlights.innerHTML = highlights
    .map(
      (item) => `
      <article class="highlight-card">
        <p class="highlight-label">${safeText(item.label)}</p>
        <p class="highlight-value">${safeText(item.value)}</p>
      </article>
    `,
    )
    .join("");
}

function renderTeamDetail(snapshot) {
  const team = state.selectedTeam ? findTeam(snapshot, state.selectedTeam) : null;

  if (!team) {
    detailTeamTitle.textContent = "Select a Team";
    teamHighlights.innerHTML = "";
    detailStatsGrid.innerHTML = "<p>No team selected.</p>";
    return;
  }

  detailTeamTitle.textContent = `${team.team} Detailed Metrics`;
  renderTeamHighlights(team);

  const statFilterText = state.statFilterText.trim().toLowerCase();
  const liveCatalog = (snapshot.stat_catalog?.live ?? []).map((entry) => ({ ...entry, source: "Live Model", sourceKind: "live" }));
  const apiCatalog = (snapshot.stat_catalog?.mlb_api ?? []).map((entry) => ({ ...entry, source: "MLB Stats API", sourceKind: "mlb_api" }));
  const workbookCatalog = (snapshot.stat_catalog?.workbook ?? []).map((entry) => ({
    ...entry,
    source: `Workbook (${entry.sheet ?? "Mscore"})`,
    sourceKind: "workbook",
  }));
  const combined = [...liveCatalog, ...apiCatalog, ...workbookCatalog];

  const filtered = combined.filter((entry) => {
    if (!statFilterText) {
      return true;
    }
    return normalizeText(entry.label).includes(statFilterText) || normalizeText(entry.group).includes(statFilterText);
  });

  const grouped = new Map();
  filtered.forEach((entry) => {
    const groupName = entry.group || "Other";
    if (!grouped.has(groupName)) {
      grouped.set(groupName, []);
    }

    let rawValue = null;
    if (entry.sourceKind === "live") {
      rawValue = team.live_stats?.[entry.key];
    } else if (entry.sourceKind === "mlb_api") {
      rawValue = team.mlb_api_stats?.[entry.key];
    } else {
      rawValue = team.workbook_stats?.[entry.key];
    }

    grouped.get(groupName).push({ ...entry, rawValue, id: statId(entry.sourceKind, entry.key) });
  });

  if (!filtered.length) {
    detailStatsGrid.innerHTML = "<p>No stats match this filter.</p>";
    return;
  }

  const groupOrder = [
    "Live Model",
    "Live Results",
    "Projection",
    "Core",
    "Offense & Series",
    "Pitching & Run Prevention",
    "Defense & Framing",
    "Model Signals",
    "Other",
  ];

  const orderedGroups = [...grouped.keys()].sort((a, b) => {
    const aIdx = groupOrder.indexOf(a);
    const bIdx = groupOrder.indexOf(b);
    const left = aIdx === -1 ? 999 : aIdx;
    const right = bIdx === -1 ? 999 : bIdx;
    return left - right;
  });

  detailStatsGrid.innerHTML = orderedGroups
    .map((groupName) => {
      const entries = grouped.get(groupName) ?? [];
      return `
      <section class="stat-group">
        <h4>${safeText(groupName)}</h4>
        <div class="group-grid">
          ${entries
            .map(
              (entry) => `
            <article class="detail-stat-card">
              <p class="detail-stat-label" title="${safeText(entry.description)}" data-stat-id="${safeText(entry.id)}">${safeText(entry.label)}</p>
              <p class="detail-stat-value">${safeText(formatMetricValue(entry.label, entry.rawValue))}</p>
              <p class="detail-stat-source">${safeText(entry.source)}</p>
            </article>
          `,
            )
            .join("")}
        </div>
      </section>
      `;
    })
    .join("");
}


function renderDiagReliabilityCurve(calibration) {
  if (!diagReliabilitySvg) {
    return;
  }

  const bins = (calibration?.reliability_curve ?? []).filter((row) => Number(row.count ?? 0) > 0);
  if (!bins.length) {
    diagReliabilitySvg.innerHTML = `<text x="20" y="40" fill="#5a7186" font-size="14">No calibration bins available yet.</text>`;
    return;
  }

  const width = 720;
  const height = 320;
  const pad = { left: 52, right: 18, top: 16, bottom: 38 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const sx = (v) => pad.left + Number(v) * plotW;
  const sy = (v) => pad.top + (1 - Number(v)) * plotH;

  const ticks = [0, 0.25, 0.5, 0.75, 1.0];
  const grid = ticks
    .map((t) => {
      const x = sx(t);
      const y = sy(t);
      const xLabel = x.toFixed(2);
      const yLabel = y.toFixed(2);
      return `
        <line x1="${xLabel}" y1="${pad.top}" x2="${xLabel}" y2="${(pad.top + plotH).toFixed(2)}" stroke="#e7eef5" stroke-width="1" />
        <line x1="${pad.left}" y1="${yLabel}" x2="${(pad.left + plotW).toFixed(2)}" y2="${yLabel}" stroke="#e7eef5" stroke-width="1" />
        <text x="${xLabel}" y="${(pad.top + plotH + 18).toFixed(2)}" fill="#5e7488" font-size="10" text-anchor="middle">${(t * 100).toFixed(0)}%</text>
        <text x="${(pad.left - 8).toFixed(2)}" y="${(y + 3).toFixed(2)}" fill="#5e7488" font-size="10" text-anchor="end">${(t * 100).toFixed(0)}%</text>
      `;
    })
    .join("");

  const points = bins.map((row) => ({
    x: Number(row.avg_pred ?? 0),
    y: Number(row.observed ?? 0),
    n: Number(row.count ?? 0),
    gap: Number(row.gap ?? 0),
  }));

  const linePath = points
    .map((pt, idx) => `${idx === 0 ? "M" : "L"} ${sx(pt.x).toFixed(2)} ${sy(pt.y).toFixed(2)}`)
    .join(" ");

  const pointsMarkup = points
    .map((pt) => {
      const r = Math.max(3.8, Math.min(8.5, 2.6 + Math.sqrt(Math.max(1, pt.n)) * 0.42));
      const color = Math.abs(pt.gap) >= 0.08 ? "#d26b2f" : "#1a7f8d";
      return `
        <g>
          <circle cx="${sx(pt.x).toFixed(2)}" cy="${sy(pt.y).toFixed(2)}" r="${r.toFixed(2)}" fill="${color}" fill-opacity="0.85" stroke="#ffffff" stroke-width="1.1">
            <title>Pred ${(pt.x * 100).toFixed(1)}% | Obs ${(pt.y * 100).toFixed(1)}% | n=${pt.n}</title>
          </circle>
        </g>
      `;
    })
    .join("");

  diagReliabilitySvg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="#f9fcff" />
    ${grid}
    <line x1="${pad.left}" y1="${(pad.top + plotH).toFixed(2)}" x2="${(pad.left + plotW).toFixed(2)}" y2="${pad.top}" stroke="#8397aa" stroke-width="1.4" stroke-dasharray="5 4" />
    <path d="${linePath}" fill="none" stroke="#156e7b" stroke-width="2.2" />
    ${pointsMarkup}
    <rect x="${pad.left}" y="${pad.top}" width="${plotW}" height="${plotH}" fill="none" stroke="#c9d8e6" stroke-width="1" />
    <text x="${(pad.left + plotW / 2).toFixed(2)}" y="${(height - 8).toFixed(2)}" fill="#33516b" font-size="11" text-anchor="middle" font-weight="600">Predicted Win Probability</text>
    <text x="14" y="${(pad.top + plotH / 2).toFixed(2)}" fill="#33516b" font-size="11" text-anchor="middle" font-weight="600" transform="rotate(-90 14 ${(pad.top + plotH / 2).toFixed(2)})">Observed Win Rate</text>
  `;
}


function renderModelDiagnostics(snapshot) {
  const diagnostics = snapshot.model_diagnostics ?? {};
  const winModel = diagnostics.win_model ?? {};
  const mscoreModel = diagnostics.mscore_model ?? {};
  const archive = diagnostics.archive_tracking ?? {};
  const marketClv = diagnostics.market_clv ?? {};

  const winMetrics = winModel.metrics_test ?? {};
  const cvMetrics = winModel.rolling_cv_mean ?? {};
  const latestArchive = archive.latest ?? {};
  const clvEdge = marketClv.avg_true_clv ?? marketClv.avg_edge_pick;
  const calibrationTest = winModel.calibration_test ?? {};
  const brierDecomp = calibrationTest.brier_decomposition ?? {};
  const driftMonitor = winModel.drift_monitor ?? {};
  const pregameSummary = winModel.pregame_context_summary ?? {};
  const blendSummary = winModel.market_blend_summary ?? {};

  const driftStatusRaw = String(driftMonitor.status ?? "").replaceAll("_", " ").trim();
  const driftStatus = driftStatusRaw ? driftStatusRaw.toUpperCase() : "-";
  const pregameReadiness = pregameSummary.readiness_pct;

  const summaryCards = [
    {
      label: "Win Model Test Acc",
      value: winMetrics.accuracy !== undefined ? formatPercent(winMetrics.accuracy, 1) : "-",
    },
    {
      label: "Win Model CV Acc",
      value: cvMetrics.accuracy !== undefined ? formatPercent(cvMetrics.accuracy, 1) : "-",
    },
    {
      label: "Drift Risk",
      value: driftStatus,
    },
    {
      label: "Overfit Index",
      value: driftMonitor.overfit_index !== null && driftMonitor.overfit_index !== undefined
        ? formatPercent(driftMonitor.overfit_index, 1)
        : "-",
    },
    {
      label: "Pregame Ready",
      value: pregameReadiness !== null && pregameReadiness !== undefined ? `${formatNumber(pregameReadiness, 1)}%` : "-",
    },
    {
      label: "Avg Market Weight",
      value: blendSummary.avg_market_weight !== null && blendSummary.avg_market_weight !== undefined
        ? formatPercent(blendSummary.avg_market_weight, 1)
        : "-",
    },
    {
      label: "Market Rows (30d)",
      value: marketClv.games_with_market ?? 0,
    },
    {
      label: marketClv.mode === "true" ? "Avg True CLV" : "Avg CLV Proxy",
      value: clvEdge !== null && clvEdge !== undefined ? formatSignedPctPoints(clvEdge) : "-",
    },
    {
      label: "Archived Days",
      value: archive.archive_count ?? 0,
    },
    {
      label: "Latest Archive",
      value: latestArchive.reference_date ?? "-",
    },
  ];

  diagSummary.innerHTML = summaryCards
    .map(
      (item) => `
      <article class="diag-pill">
        <p class="diag-pill-label">${safeText(item.label)}</p>
        <p class="diag-pill-value">${safeText(item.value)}</p>
      </article>
      `,
    )
    .join("");

  const winImportanceRows = (winModel.feature_importance ?? [])
    .slice(0, 8)
    .map(
      (row) => `
      <div class="diag-row">
        <div class="diag-row-top">
          <span class="diag-row-title">${safeText(row.feature)}</span>
          <span class="diag-row-value">${safeText(formatNumber(row.importance_pct, 1))}%</span>
        </div>
        <p class="diag-row-sub">Coef ${safeText(formatNumber(row.coefficient, 4))} (${safeText(row.direction ?? "-")})</p>
      </div>
      `,
    )
    .join("");
  diagWinImportance.innerHTML = winImportanceRows || "<p class='muted-note'>No feature-importance data yet.</p>";

  const winAblationRows = (winModel.ablation?.rows ?? [])
    .slice(0, 8)
    .map(
      (row) => `
      <div class="diag-row">
        <div class="diag-row-top">
          <span class="diag-row-title">${safeText(row.feature)}</span>
          <span class="diag-row-value">+${safeText(formatNumber(row.delta_log_loss, 3))} LogLoss</span>
        </div>
        <p class="diag-row-sub">Acc delta ${safeText(formatPercent(row.delta_accuracy, 1))}</p>
      </div>
      `,
    )
    .join("");
  diagWinAblation.innerHTML = winAblationRows || "<p class='muted-note'>No ablation data yet.</p>";

  if (diagClvList) {
    const modeLabel = marketClv.mode === "true" ? "True CLV" : "CLV Proxy";
    const clvRows = [
      {
        title: "Mode",
        value: modeLabel,
        sub: marketClv.mode === "true"
          ? "Using opening/closing probabilities for pick-side CLV."
          : "Using model-vs-market edge as CLV proxy until true close lines are present.",
      },
      {
        title: "Finalized Accuracy",
        value: marketClv.finalized_accuracy !== null && marketClv.finalized_accuracy !== undefined
          ? formatPercent(marketClv.finalized_accuracy, 1)
          : "-",
        sub: `Finalized with market: ${safeText(marketClv.finalized_with_market ?? 0)}`,
      },
      {
        title: "Strong Edge Accuracy",
        value: marketClv.strong_edge_accuracy !== null && marketClv.strong_edge_accuracy !== undefined
          ? formatPercent(marketClv.strong_edge_accuracy, 1)
          : "-",
        sub: `|Edge| >= 5 pts sample: ${safeText(marketClv.strong_edge_games ?? 0)}`,
      },
      {
        title: "Positive Edge Rate",
        value: marketClv.positive_edge_rate !== null && marketClv.positive_edge_rate !== undefined
          ? formatPercent(marketClv.positive_edge_rate, 1)
          : "-",
        sub: `Days analyzed: ${safeText(marketClv.days_analyzed ?? 0)}`,
      },
    ];

    const headerRows = clvRows
      .map(
        (row) => `
        <div class="diag-row">
          <div class="diag-row-top">
            <span class="diag-row-title">${safeText(row.title)}</span>
            <span class="diag-row-value">${safeText(row.value)}</span>
          </div>
          <p class="diag-row-sub">${safeText(row.sub)}</p>
        </div>
        `,
      )
      .join("");

    const dailyRows = (marketClv.daily ?? [])
      .slice(0, 7)
      .map(
        (row) => {
          const dayEdge = row.avg_true_clv ?? row.avg_edge_pick;
          return `
          <div class="diag-row">
            <div class="diag-row-top">
              <span class="diag-row-title">${safeText(row.reference_date ?? "-")}</span>
              <span class="diag-row-value">${safeText(dayEdge !== null && dayEdge !== undefined ? formatSignedPctPoints(dayEdge) : "-")}</span>
            </div>
            <p class="diag-row-sub">Games ${safeText(row.games_with_market ?? 0)} | Acc ${safeText(row.accuracy !== null && row.accuracy !== undefined ? formatPercent(row.accuracy, 1) : "-")}</p>
          </div>
          `;
        },
      )
      .join("");

    diagClvList.innerHTML = (headerRows + dailyRows) || "<p class='muted-note'>No market diagnostics available yet.</p>";
  }

  if (diagCalibrationSummary) {
    const ece = calibrationTest.ece;
    const mce = calibrationTest.mce;
    const sampleSize = calibrationTest.sample_size;
    const baseRate = calibrationTest.base_rate;
    const binsUsed = (calibrationTest.reliability_curve ?? []).filter((row) => Number(row.count ?? 0) > 0).length;

    const calRows = [
      {
        title: "Calibration ECE",
        value: ece === null || ece === undefined ? "-" : `${(Number(ece) * 100).toFixed(1)}%`,
        sub: "Lower is better. Mean absolute gap between predicted and observed win rates.",
      },
      {
        title: "Calibration MCE",
        value: mce === null || mce === undefined ? "-" : `${(Number(mce) * 100).toFixed(1)}%`,
        sub: "Worst-bin calibration gap.",
      },
      {
        title: "Calibration Sample",
        value: sampleSize === null || sampleSize === undefined ? "-" : String(sampleSize),
        sub: `Backtest bins used: ${safeText(String(binsUsed || 0))} | Base rate ${baseRate === null || baseRate === undefined ? "-" : formatPercent(baseRate, 1)}`,
      },
      {
        title: "Drift Monitor",
        value: driftStatus,
        sub: `Overfit ${driftMonitor.overfit_index !== null && driftMonitor.overfit_index !== undefined ? formatPercent(driftMonitor.overfit_index, 1) : "-"} | CV std ${driftMonitor.cv_accuracy_std !== null && driftMonitor.cv_accuracy_std !== undefined ? formatPercent(driftMonitor.cv_accuracy_std, 1) : "-"}`,
      },
      {
        title: "Pregame Gating",
        value: pregameReadiness !== null && pregameReadiness !== undefined ? `${formatNumber(pregameReadiness, 1)}%` : "-",
        sub: `Low-readiness games today: ${safeText(String(pregameSummary.low_readiness_games ?? 0))}`,
      },
      {
        title: "Market Blend Avg",
        value: blendSummary.avg_market_weight !== null && blendSummary.avg_market_weight !== undefined ? formatPercent(blendSummary.avg_market_weight, 1) : "-",
        sub: `Games with market lines: ${safeText(String(blendSummary.games_with_market ?? 0))}`,
      },
    ];

    diagCalibrationSummary.innerHTML = calRows
      .map(
        (row) => `
        <div class="diag-row">
          <div class="diag-row-top">
            <span class="diag-row-title">${safeText(row.title)}</span>
            <span class="diag-row-value">${safeText(row.value)}</span>
          </div>
          <p class="diag-row-sub">${safeText(row.sub)}</p>
        </div>
        `,
      )
      .join("");
  }

  renderDiagReliabilityCurve(calibrationTest);

  if (diagBrierDecomp) {
    const decompRows = [
      {
        title: "Brier Score",
        value: brierDecomp.brier_score === null || brierDecomp.brier_score === undefined ? "-" : formatNumber(brierDecomp.brier_score, 4),
        sub: "Overall probability error (lower is better).",
      },
      {
        title: "Reliability",
        value: brierDecomp.reliability === null || brierDecomp.reliability === undefined ? "-" : formatNumber(brierDecomp.reliability, 4),
        sub: "Calibration penalty term (lower is better).",
      },
      {
        title: "Resolution",
        value: brierDecomp.resolution === null || brierDecomp.resolution === undefined ? "-" : formatNumber(brierDecomp.resolution, 4),
        sub: "Separation power term (higher is better).",
      },
      {
        title: "Uncertainty",
        value: brierDecomp.uncertainty === null || brierDecomp.uncertainty === undefined ? "-" : formatNumber(brierDecomp.uncertainty, 4),
        sub: "Intrinsic baseline variance from class balance.",
      },
    ];

    diagBrierDecomp.innerHTML = decompRows
      .map(
        (row) => `
        <div class="diag-row">
          <div class="diag-row-top">
            <span class="diag-row-title">${safeText(row.title)}</span>
            <span class="diag-row-value">${safeText(row.value)}</span>
          </div>
          <p class="diag-row-sub">${safeText(row.sub)}</p>
        </div>
        `,
      )
      .join("");
  }

  const mscoreImportanceRows = (mscoreModel.components ?? [])
    .slice(0, 8)
    .map(
      (row) => `
      <div class="diag-row">
        <div class="diag-row-top">
          <span class="diag-row-title">${safeText(row.label)}</span>
          <span class="diag-row-value">${safeText(formatNumber(row.weight_pct, 1))}%</span>
        </div>
        <p class="diag-row-sub">Avg points ${safeText(formatNumber(row.avg_points, 2))}</p>
      </div>
      `,
    )
    .join("");
  diagMscoreImportance.innerHTML = mscoreImportanceRows || "<p class='muted-note'>No Mscore diagnostics yet.</p>";

  const mscoreAblationRows = (mscoreModel.ablation ?? [])
    .slice(0, 8)
    .map(
      (row) => `
      <div class="diag-row">
        <div class="diag-row-top">
          <span class="diag-row-title">${safeText(row.label)}</span>
          <span class="diag-row-value">Shift ${safeText(formatNumber(row.mean_abs_rank_shift, 2))}</span>
        </div>
        <p class="diag-row-sub">Top5 turnover ${safeText(row.top5_turnover)} | Max shift ${safeText(row.max_rank_shift)}</p>
      </div>
      `,
    )
    .join("");
  diagMscoreAblation.innerHTML = mscoreAblationRows || "<p class='muted-note'>No Mscore ablation data yet.</p>";
}

function renderAll(snapshot) {
  ensureSelectedTeam(snapshot);
  renderSummary(snapshot);
  renderTeams(snapshot);
  renderRankingHeaderHints(snapshot);
  renderGames(snapshot);
  renderPredictions(snapshot);
  renderValueFinder(snapshot);
  renderTopChart(snapshot);
  renderModelDiagnostics(snapshot);
  renderAdvancedPanel(snapshot);
  renderTeamDetail(snapshot);
  renderStatRankingSidebar(snapshot);
  renderMatchupSidebar(snapshot);

  const generatedAt = snapshot.meta?.generated_at;
  const generatedText = generatedAt ? new Date(generatedAt).toLocaleString() : "Unknown";
  refreshTime.textContent = `Last updated: ${generatedText}`;
}

function setError(message) {
  if (!message) {
    errorBox.classList.add("hidden");
    errorBox.textContent = "";
    return;
  }
  errorBox.classList.remove("hidden");
  errorBox.textContent = message;
}

async function loadDashboard(forceRefresh = false) {
  refreshBtn.disabled = true;

  try {
    const response = await fetch(forceRefresh ? "/api/refresh" : "/api/dashboard", {
      method: forceRefresh ? "POST" : "GET",
    });
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const snapshot = await response.json();
    state.snapshot = snapshot;
    renderAll(snapshot);
    await loadRrgData();
    setError("");
  } catch (error) {
    setError(`Could not load dashboard data: ${error.message}`);
  } finally {
    refreshBtn.disabled = false;
  }
}

teamFilter.addEventListener("input", () => {
  state.filterText = teamFilter.value;
  if (state.snapshot) {
    renderTeams(state.snapshot);
    renderStatRankingSidebar(state.snapshot);
  }
});

statFilter.addEventListener("input", () => {
  state.statFilterText = statFilter.value;
  if (state.snapshot) {
    renderTeamDetail(state.snapshot);
    renderStatRankingSidebar(state.snapshot);
  }
});

teamsBody.addEventListener("click", (event) => {
  const row = event.target.closest("tr[data-team]");
  if (!row || !state.snapshot) {
    return;
  }

  state.selectedTeam = row.getAttribute("data-team");
  renderTeams(state.snapshot);
  renderTeamDetail(state.snapshot);
  renderAdvancedPanel(state.snapshot);
  renderStatRankingSidebar(state.snapshot);
});

rankingHead.addEventListener("click", (event) => {
  const header = event.target.closest("th[data-stat-id]");
  if (!header || !state.snapshot) {
    return;
  }

  closeMatchupSidebar();
  openStatSidebar(state.snapshot, header.getAttribute("data-stat-id"));
});

detailStatsGrid.addEventListener("click", (event) => {
  const label = event.target.closest(".detail-stat-label[data-stat-id]");
  if (!label || !state.snapshot) {
    return;
  }

  closeMatchupSidebar();
  openStatSidebar(state.snapshot, label.getAttribute("data-stat-id"));
});

statSidebarClose.addEventListener("click", () => {
  closeStatSidebar();
});

matchupSidebarClose.addEventListener("click", () => {
  closeMatchupSidebar();
});

matchupExportPngBtn?.addEventListener("click", () => {
  exportMatchupBreakdownPng(state.snapshot);
});

predictionBars.addEventListener("click", (event) => {
  if (!state.snapshot) {
    return;
  }

  const compareBtn = event.target.closest("button[data-sp-compare]");
  if (compareBtn) {
    event.preventDefault();
    event.stopPropagation();
    toggleStarterComparison(compareBtn.getAttribute("data-matchup-id"));
    return;
  }

  if (event.target.closest(".sp-compare-panel")) {
    return;
  }

  const row = event.target.closest(".bar-row[data-matchup-id]");
  if (!row) {
    return;
  }

  openMatchupSidebar(state.snapshot, row.getAttribute("data-matchup-id"));
});

predictionBars.addEventListener("keydown", (event) => {
  if (event.target.closest(".sp-compare-btn") || event.target.closest(".sp-compare-panel")) {
    return;
  }

  const row = event.target.closest(".bar-row[data-matchup-id]");
  if (!row || !state.snapshot) {
    return;
  }

  if (event.key !== "Enter" && event.key !== " ") {
    return;
  }

  event.preventDefault();
  openMatchupSidebar(state.snapshot, row.getAttribute("data-matchup-id"));
});

valueFinderList?.addEventListener("click", (event) => {
  const row = event.target.closest(".value-row[data-matchup-id]");
  if (!row || !state.snapshot) {
    return;
  }

  openMatchupSidebar(state.snapshot, row.getAttribute("data-matchup-id"));
});

valueFinderList?.addEventListener("keydown", (event) => {
  const row = event.target.closest(".value-row[data-matchup-id]");
  if (!row || !state.snapshot) {
    return;
  }

  if (event.key !== "Enter" && event.key !== " ") {
    return;
  }

  event.preventDefault();
  openMatchupSidebar(state.snapshot, row.getAttribute("data-matchup-id"));
});

rrgMetricSelect?.addEventListener("change", () => {
  state.rrgMetric = rrgMetricSelect.value;
  loadRrgData();
});

rrgLookback?.addEventListener("input", () => {
  state.rrgLookback = Number(rrgLookback.value);
  updateRrgLabels();
});

rrgLookback?.addEventListener("change", () => {
  state.rrgLookback = Number(rrgLookback.value);
  loadRrgData();
});

rrgTrailRange?.addEventListener("change", () => {
  state.rrgTrail = Number(rrgTrailRange.value);
  loadRrgData();
});

rrgSizeSelect?.addEventListener("change", () => {
  state.rrgSize = Number(rrgSizeSelect.value);
  applyRrgChartSize();
});

rrgYMinInput?.addEventListener("change", () => {
  state.rrgYMin = parseOptionalNumber(rrgYMinInput.value);
  if (state.rrgPayload) {
    renderRrg(state.rrgPayload);
  }
});

rrgYMaxInput?.addEventListener("change", () => {
  state.rrgYMax = parseOptionalNumber(rrgYMaxInput.value);
  if (state.rrgPayload) {
    renderRrg(state.rrgPayload);
  }
});

rrgShowTrails?.addEventListener("change", () => {
  state.rrgShowTrails = Boolean(rrgShowTrails.checked);
  if (state.rrgPayload) {
    renderRrg(state.rrgPayload);
  }
});

rrgExportPngBtn?.addEventListener("click", () => {
  exportRrgAsPng();
});

rrgMomentumExportPngBtn?.addEventListener("click", () => {
  exportRrgMomentumSheetAsPng();
});

advancedExportPngBtn?.addEventListener("click", () => {
  exportAdvancedScatterAsPng();
});

rrgTeamToggle?.addEventListener("click", (event) => {
  event.stopPropagation();
  setRrgTeamMenuOpen(!state.rrgTeamMenuOpen);
});

rrgTeamAllBtn?.addEventListener("click", () => {
  state.rrgSelectedTeams = [...(state.rrgTeamUniverse ?? [])];
  renderRrgTeamSelector();
  if (state.rrgPayload) {
    renderRrg(state.rrgPayload);
  }
});

rrgTeamNoneBtn?.addEventListener("click", () => {
  state.rrgSelectedTeams = [];
  renderRrgTeamSelector();
  if (state.rrgPayload) {
    renderRrg(state.rrgPayload);
  }
});

rrgTeamOptions?.addEventListener("change", (event) => {
  const target = event.target.closest("input[data-team]");
  if (!target) {
    return;
  }

  const team = target.getAttribute("data-team") || "";
  const selected = new Set(state.rrgSelectedTeams ?? []);
  if (target.checked) {
    selected.add(team);
  } else {
    selected.delete(team);
  }
  state.rrgSelectedTeams = [...selected];
  renderRrgTeamSelector();

  if (state.rrgPayload) {
    renderRrg(state.rrgPayload);
  }
});

document.addEventListener("click", (event) => {
  if (!rrgTeamPicker) {
    return;
  }
  if (rrgTeamPicker.contains(event.target)) {
    return;
  }
  setRrgTeamMenuOpen(false);
});

refreshBtn.addEventListener("click", () => {
  loadDashboard(true);
});

applyRrgChartSize();

loadDashboard(false);
setInterval(() => loadDashboard(false), 60000);
