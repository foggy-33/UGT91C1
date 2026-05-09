const sourceSelect = document.querySelector("#sourceSelect");
const toast = document.querySelector("#toast");

const charts = {
  top: document.querySelector("#topChart"),
  hist: document.querySelector("#histChart"),
  position: document.querySelector("#positionChart"),
  support: document.querySelector("#supportChart"),
};

function ctx(name) {
  return charts[name].getContext("2d");
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  window.setTimeout(() => toast.classList.add("hidden"), 3600);
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function clear(canvas, context) {
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#fff";
  context.fillRect(0, 0, canvas.width, canvas.height);
}

function fitText(context, text, maxWidth) {
  let value = String(text || "-");
  if (context.measureText(value).width <= maxWidth) return value;
  while (value.length > 4 && context.measureText(`${value}...`).width > maxWidth) {
    value = value.slice(0, -1);
  }
  return `${value}...`;
}

function drawBars(canvas, context, labels, values, options = {}) {
  clear(canvas, context);
  if (!values.length) return drawEmpty(canvas, context, "暂无数据");

  const left = options.left || 180;
  const top = options.top || 28;
  const right = 58;
  const rowH = options.rowH || 18;
  const gap = options.gap || 11;
  const max = Math.max(...values, 0.01);
  const barMax = canvas.width - left - right;
  const labelMax = left - 28;

  context.font = "12px Segoe UI, Microsoft YaHei";
  values.forEach((value, index) => {
    const y = top + index * (rowH + gap);
    const barW = Math.max(3, (value / max) * barMax);
    const gradient = context.createLinearGradient(left, y, left + barW, y);
    gradient.addColorStop(0, "#2563eb");
    gradient.addColorStop(1, "#06b6d4");
    context.fillStyle = gradient;
    context.fillRect(left, y, barW, rowH);
    context.fillStyle = "#334155";
    context.fillText(fitText(context, labels[index], labelMax), 14, y + rowH - 4);
    context.fillText(fmt(value), left + barW + 8, y + rowH - 4);
  });
}

function drawVerticalBars(canvas, context, labels, values) {
  clear(canvas, context);
  if (!values.length) return drawEmpty(canvas, context, "暂无数据");

  const left = 42;
  const right = 22;
  const top = 26;
  const bottom = 54;
  const max = Math.max(...values, 1);
  const plotW = canvas.width - left - right;
  const plotH = canvas.height - top - bottom;
  const barW = Math.max(8, plotW / values.length - 8);

  context.font = "11px Segoe UI, Microsoft YaHei";
  values.forEach((value, index) => {
    const x = left + index * (plotW / values.length) + 3;
    const h = (value / max) * plotH;
    const y = top + plotH - h;
    context.fillStyle = "#2563eb";
    context.fillRect(x, y, barW, h);
    context.fillStyle = "#64748b";
    context.save();
    context.translate(x + 2, canvas.height - 18);
    context.rotate(-0.55);
    context.fillText(String(labels[index]).slice(0, 12), 0, 0);
    context.restore();
  });
}

function drawScatter(canvas, context, points) {
  clear(canvas, context);
  if (!points.length) return drawEmpty(canvas, context, "当前结果缺少家族支持字段");

  const left = 56;
  const right = 26;
  const top = 24;
  const bottom = 46;
  const plotW = canvas.width - left - right;
  const plotH = canvas.height - top - bottom;
  const maxX = Math.max(...points.map((p) => p.support), 1);
  const minY = Math.min(...points.map((p) => p.score));
  const maxY = Math.max(...points.map((p) => p.score));
  const yRange = Math.max(maxY - minY, 0.01);

  context.strokeStyle = "#d8e5f7";
  context.beginPath();
  context.moveTo(left, top);
  context.lineTo(left, top + plotH);
  context.lineTo(left + plotW, top + plotH);
  context.stroke();

  points.forEach((point) => {
    const x = left + (point.support / maxX) * plotW;
    const y = top + plotH - ((point.score - minY) / yRange) * plotH;
    context.fillStyle = "rgba(37, 99, 235, 0.58)";
    context.beginPath();
    context.arc(x, y, 4, 0, Math.PI * 2);
    context.fill();
  });

  context.fillStyle = "#64748b";
  context.font = "12px Segoe UI, Microsoft YaHei";
  context.fillText("家族支持", left + plotW - 56, canvas.height - 14);
  context.save();
  context.translate(18, top + 80);
  context.rotate(-Math.PI / 2);
  context.fillText("预测值", 0, 0);
  context.restore();
}

function drawEmpty(canvas, context, text) {
  clear(canvas, context);
  context.fillStyle = "#64748b";
  context.font = "15px Segoe UI, Microsoft YaHei";
  context.fillText(text, 28, 70);
}

function renderTable(top) {
  document.querySelector("#topRows").innerHTML = top.map((item) => `
    <tr>
      <td>${item.rank}</td>
      <td><strong>${item.variant || "-"}</strong></td>
      <td>${fmt(item.score)}</td>
      <td>${fmt(item.family_support, 2)}</td>
      <td>${item.source || "-"}</td>
    </tr>
  `).join("");
}

async function loadSummary() {
  const response = await fetch(`/api/results/summary?source=${sourceSelect.value}`);
  const data = await response.json();
  if (!response.ok) {
    showToast(data.error || "结果读取失败。");
    return;
  }

  document.querySelector("#vizTotal").textContent = data.total;
  document.querySelector("#vizTop").textContent = fmt(data.top_score);
  document.querySelector("#vizMedian").textContent = fmt(data.median_score);
  document.querySelector("#vizMean").textContent = fmt(data.mean_score);
  document.querySelector("#scoreLabel").textContent = data.score_col;
  document.querySelector("#fileName").textContent = data.file;

  drawBars(charts.top, ctx("top"), data.top.map((x) => x.variant), data.top.map((x) => x.score), { left: 230 });
  drawVerticalBars(charts.hist, ctx("hist"), data.histogram.map((x) => x.label), data.histogram.map((x) => x.count));
  drawVerticalBars(charts.position, ctx("position"), data.positions.map((x) => x.position), data.positions.map((x) => x.count));
  drawScatter(charts.support, ctx("support"), data.support_points);
  renderTable(data.top);
}

sourceSelect.addEventListener("change", loadSummary);
loadSummary();
