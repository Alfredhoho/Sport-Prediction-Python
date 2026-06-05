const apiBaseInput = document.querySelector("#apiBase");
const loadTeamsBtn = document.querySelector("#loadTeamsBtn");
const predictBtn = document.querySelector("#predictBtn");
const homeTeamSelect = document.querySelector("#homeTeam");
const awayTeamSelect = document.querySelector("#awayTeam");
const includeAI = document.querySelector("#includeAI");
const statusEl = document.querySelector("#status");
const resultsEl = document.querySelector("#results");

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function percent(decimal) {
  return `${Math.round(decimal * 1000) / 10}%`;
}

function apiBase() {
  return apiBaseInput.value.replace(/\/$/, "");
}

function fillTeamSelect(select, teams) {
  select.innerHTML = "";
  for (const team of teams) {
    const option = document.createElement("option");
    option.value = team;
    option.textContent = team;
    select.appendChild(option);
  }
}

async function loadTeams() {
  setStatus("Loading teams...");
  try {
    const response = await fetch(`${apiBase()}/teams`);
    if (!response.ok) throw new Error(`Backend returned ${response.status}`);
    const data = await response.json();
    fillTeamSelect(homeTeamSelect, data.teams);
    fillTeamSelect(awayTeamSelect, data.teams);
    if (data.teams.length > 1) awayTeamSelect.selectedIndex = 1;
    setStatus(`Loaded ${data.teams.length} teams from ${data.matches_loaded} matches. Half-life: ${data.half_life_days} days.`);
  } catch (error) {
    setStatus(`Could not load teams: ${error.message}`, true);
  }
}

async function predict() {
  const payload = {
    home_team: homeTeamSelect.value,
    away_team: awayTeamSelect.value,
    include_ai_explanation: includeAI.checked,
  };

  setStatus("Calculating form-weighted prediction...");
  predictBtn.disabled = true;
  try {
    const response = await fetch(`${apiBase()}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Backend returned ${response.status}`);
    renderPrediction(data);
    setStatus("Prediction complete. Newer matches had more influence than older matches.");
  } catch (error) {
    setStatus(`Prediction failed: ${error.message}`, true);
  } finally {
    predictBtn.disabled = false;
  }
}

function renderPrediction(data) {
  resultsEl.classList.remove("hidden");
  document.querySelector("#matchTitle").textContent = `${data.home_team} vs ${data.away_team}`;
  document.querySelector("#homeWinLabel").textContent = `${data.home_team} win`;
  document.querySelector("#awayWinLabel").textContent = `${data.away_team} win`;
  document.querySelector("#homeWin").textContent = percent(data.probabilities.home_win);
  document.querySelector("#draw").textContent = percent(data.probabilities.draw);
  document.querySelector("#awayWin").textContent = percent(data.probabilities.away_win);
  document.querySelector("#expectedGoals").textContent =
    `${data.home_team}: ${data.expected_goals.home} | ${data.away_team}: ${data.expected_goals.away}`;

  const modelInfo = document.querySelector("#modelInfo");
  modelInfo.innerHTML = "";
  const facts = [
    `Method: ${data.model_info.method}`,
    `Matches used: ${data.model_info.matches_used}`,
    `Latest match date: ${data.model_info.latest_match_date}`,
    `Recency half-life: ${data.model_info.half_life_days} days`,
    `${data.home_team}: ${data.model_info.home_team_matches} raw matches, ${data.model_info.home_team_weighted_matches} weighted`,
    `${data.away_team}: ${data.model_info.away_team_matches} raw matches, ${data.model_info.away_team_weighted_matches} weighted`,
    `Confidence score: ${data.model_info.confidence}`,
  ];
  for (const fact of facts) {
    const li = document.createElement("li");
    li.textContent = fact;
    modelInfo.appendChild(li);
  }

  const scoreBars = document.querySelector("#scoreBars");
  scoreBars.innerHTML = "";
  const maxProb = Math.max(...data.top_scores.map((item) => item.probability));
  for (const item of data.top_scores) {
    const row = document.createElement("div");
    row.className = "score-row";
    row.innerHTML = `
      <strong>${item.score}</strong>
      <div class="bar"><div class="bar-fill" style="width: ${(item.probability / maxProb) * 100}%"></div></div>
      <span>${percent(item.probability)}</span>
    `;
    scoreBars.appendChild(row);
  }

  renderAI(data.ai_explanation);
}

function renderAI(explanation) {
  const aiCard = document.querySelector(".ai-card");
  if (!explanation) {
    aiCard.classList.add("hidden");
    return;
  }
  aiCard.classList.remove("hidden");
  document.querySelector("#aiHeadline").textContent = explanation.headline;
  document.querySelector("#aiPlain").textContent = explanation.plain_english;
  document.querySelector("#aiCaution").textContent = explanation.caution;
  const list = document.querySelector("#aiFactors");
  list.innerHTML = "";
  for (const factor of explanation.key_factors) {
    const li = document.createElement("li");
    li.textContent = factor;
    list.appendChild(li);
  }
}

loadTeamsBtn.addEventListener("click", loadTeams);
predictBtn.addEventListener("click", predict);
loadTeams();
