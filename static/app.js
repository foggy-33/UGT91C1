const form = document.querySelector("#predictForm");
const fileInput = document.querySelector("#fileInput");
const fileName = document.querySelector("#fileName");
const dropzone = document.querySelector("#dropzone");
const toast = document.querySelector("#toast");
const emptyState = document.querySelector("#emptyState");
const results = document.querySelector("#results");
const steps = [...document.querySelectorAll(".step")];
const rows = document.querySelector("#resultRows");
const scoreCanvas = document.querySelector("#scoreChart");
const scoreCtx = scoreCanvas.getContext("2d");
const foldCanvas = document.querySelector("#foldChart");
const foldCtx = foldCanvas.getContext("2d");
const workflowStatus = document.querySelector("#workflowStatus");

function showToast(message) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  window.setTimeout(() => toast.classList.add("hidden"), 3600);
}

function setStep(index, message) {
  steps.forEach((step, i) => {
    step.classList.toggle("done", i < index);
    step.classList.toggle("active", i === index);
  });
  workflowStatus.textContent = message || (index >= steps.length ? "流程完成" : `正在执行 ${String(index + 1).padStart(2, "0")}`);
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function bestLabel(item) {
  return item.activity_rel ?? item.activity ?? "-";
}

function updateFileName() {
  const file = fileInput.files[0];
  fileName.textContent = file ? file.name : "选择或拖入 CSV 文件";
}

function clearCanvas(canvas, ctx) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function fitText(ctx, text, maxWidth) {
  const raw = String(text || "-");
  if (ctx.measureText(raw).width <= maxWidth) return raw;
  let out = raw;
  while (out.length > 4 && ctx.measureText(`${out}...`).width > maxWidth) {
    out = out.slice(0, -1);
  }
  return `${out}...`;
}

function drawHorizontalBars(canvas, ctx, labels, values, options = {}) {
  clearCanvas(canvas, ctx);

  if (!values.length) {
    ctx.fillStyle = "#64748b";
    ctx.font = "16px Segoe UI, Microsoft YaHei";
    ctx.fillText(options.emptyText || "暂无数据", 28, 72);
    return;
  }

  const left = options.left || 132;
  const right = 56;
  const top = options.top || 30;
  const rowH = options.rowH || 18;
  const gap = options.gap || 12;
  const max = Math.max(...values.map((x) => Math.abs(x)), 0.01);
  const barMax = canvas.width - left - right;
  const labelMax = left - 28;

  ctx.font = "12px Segoe UI, Microsoft YaHei";
  values.forEach((value, index) => {
    const y = top + index * (rowH + gap);
    const barW = Math.max(4, (Math.abs(value) / max) * barMax);
    const gradient = ctx.createLinearGradient(left, y, left + barW, y);
    gradient.addColorStop(0, "#2563eb");
    gradient.addColorStop(1, "#06b6d4");
    ctx.fillStyle = gradient;
    ctx.fillRect(left, y, barW, rowH);
    ctx.fillStyle = "#334155";
    ctx.fillText(fitText(ctx, labels[index], labelMax), 14, y + rowH - 4);
    ctx.fillText(fmt(value), left + barW + 9, y + rowH - 4);
  });
}

function drawScoreChart(records) {
  const data = records.slice(0, 16);
  drawHorizontalBars(
    scoreCanvas,
    scoreCtx,
    data.map((item) => item.variant),
    data.map((item) => Number(item.pred_activity_family || 0)),
    { left: 220, emptyText: "暂无预测结果" }
  );
}

function drawFoldChart(evaluation) {
  if (!evaluation || !evaluation.available) {
    drawHorizontalBars(foldCanvas, foldCtx, [], [], {
      emptyText: "上传含 activity_rel 或 activity 的 CSV 后，将显示 5 折 RMSE。"
    });
    return;
  }
  drawHorizontalBars(
    foldCanvas,
    foldCtx,
    evaluation.folds.map((item) => `第 ${item.fold} 折`),
    evaluation.folds.map((item) => Number(item.rmse || 0)),
    { left: 92, rowH: 24, gap: 14 }
  );
}

function renderEvaluation(evaluation) {
  document.querySelector("#evalStatus").textContent = evaluation.message || "已完成本次五折评估";
  document.querySelector("#evalRmse").textContent = evaluation.available ? fmt(evaluation.rmse_mean) : "-";
  document.querySelector("#evalMae").textContent = evaluation.available ? fmt(evaluation.mae_mean) : "-";
  document.querySelector("#evalR2").textContent = evaluation.available ? fmt(evaluation.r2) : "-";
  document.querySelector("#evalSamples").textContent = evaluation.available ? evaluation.n_samples : "-";
  drawFoldChart(evaluation);
}

function renderTable(records) {
  rows.innerHTML = records.map((item) => `
    <tr>
      <td>${item.rank ?? "-"}</td>
      <td><strong>${item.variant ?? "-"}</strong></td>
      <td>${fmt(bestLabel(item))}</td>
      <td>${fmt(item.pred_activity_family)}</td>
      <td>${fmt(item.cv_error)}</td>
      <td>${fmt(item.family_support, 2)}</td>
      <td>${item.source ?? "-"}</td>
    </tr>
  `).join("");
}

function renderResult(data) {
  emptyState.classList.add("hidden");
  results.classList.remove("hidden");
  document.querySelector("#metricInput").textContent = data.counts.input;
  document.querySelector("#metricFeatures").textContent = data.counts.features;
  document.querySelector("#metricTop").textContent = fmt(data.summary.top_score);
  document.querySelector("#metricMedian").textContent = fmt(data.summary.median_score);
  document.querySelector("#downloadAll").href = data.downloads.all;
  document.querySelector("#reportText").textContent = data.evaluation.available
    ? "导出的 CSV 会保留原始字段，并追加五折预测活性、真实活性误差与排序结果。"
    : "导出的 CSV 会保留原始字段，并追加基于当前模型的预测活性与排序结果。";
  renderEvaluation(data.evaluation);
  renderTable(data.records);
  drawScoreChart(data.records);
}

async function pollJob(jobId) {
  while (true) {
    const response = await fetch(`/api/predict/status/${jobId}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || "任务状态获取失败。");

    setStep(job.step ?? 0, job.message);

    if (job.state === "done") {
      steps.forEach((step) => {
        step.classList.add("done");
        step.classList.remove("active");
      });
      workflowStatus.textContent = "流程完成";
      return job.result;
    }
    if (job.state === "error") {
      throw new Error(job.message || "预测失败。");
    }

    await new Promise((resolve) => window.setTimeout(resolve, 300));
  }
}

fileInput.addEventListener("change", updateFileName);

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  fileInput.files = event.dataTransfer.files;
  updateFileName();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files[0]) {
    showToast("请先选择 CSV 文件。");
    return;
  }

  const body = new FormData();
  body.append("file", fileInput.files[0]);

  const button = form.querySelector("button");
  button.disabled = true;
  button.textContent = "正在运行...";
  results.classList.add("hidden");
  emptyState.classList.remove("hidden");
  setStep(0, "上传文件");

  try {
    const startResponse = await fetch("/api/predict/start", { method: "POST", body });
    const startData = await startResponse.json();
    if (!startResponse.ok) throw new Error(startData.error || "任务启动失败。");

    const data = await pollJob(startData.job_id);
    renderResult(data);
    showToast(data.evaluation.available ? "预测完成：已基于上传数据完成五折评估。" : "预测完成：本次 CSV 无标签，仅生成活性预测排序。");
  } catch (error) {
    setStep(0, "任务失败");
    showToast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "开始预测";
  }
});

drawFoldChart(null);
