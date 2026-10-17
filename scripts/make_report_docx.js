/**
 * 生成 UGT91C1 项目报告的 Word 版本。
 *
 * 运行（需先把 docx 包安装到临时目录，避免污染仓库）：
 *   $env:NODE_PATH = "$env:TEMP\ugt91c1-docx-build\node_modules"
 *   node scripts/make_report_docx.js
 *
 * 输出: UGT91C1-突变体活性预测-项目报告.docx
 */
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, AlignmentType, LevelFormat, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageBreak,
} = require('docx');

const BASE = path.resolve(__dirname, '..');
const FIG = path.join(BASE, 'results', 'figures');
const OUT = path.join(BASE, 'UGT91C1-突变体活性预测-项目报告.docx');

const CJK = 'Microsoft YaHei';
const FONT = { ascii: 'Arial', hAnsi: 'Arial', eastAsia: CJK };

const C_BLACK = '1A1A1A';
const C_BLUE = '1F4E9C';
const C_GOLD = 'A8863A';
const C_MUTED = '595959';
const C_RULE = 'D9E1EC';
const C_HEADFILL = 'EAF0F8';
const C_CALLFILL = 'F4F7FC';

const border = { style: BorderStyle.SINGLE, size: 1, color: C_RULE };
const borders = { top: border, bottom: border, left: border, right: border };

/* ---------- 基础构造 ---------- */
function run(text, o) {
  o = o || {};
  return new TextRun({
    text: text, font: FONT, size: o.size || 21,
    bold: !!o.bold, color: o.color || C_BLACK, italics: !!o.italics,
  });
}

function P(children, o) {
  o = o || {};
  if (!Array.isArray(children)) children = [children];
  return new Paragraph({
    children: children,
    alignment: o.align,
    spacing: { before: o.before === undefined ? 60 : o.before, after: o.after === undefined ? 60 : o.after, line: o.line || 300 },
    shading: o.fill ? { fill: o.fill, type: ShadingType.CLEAR } : undefined,
    indent: o.indent,
    pageBreakBefore: !!o.pageBreakBefore,
  });
}

function T(text, o) { return P([run(text, o)], o); }

function H1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 300, after: 140 },
    children: [run(text, { size: 30, bold: true, color: C_BLACK })],
  });
}

function H2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 220, after: 100 },
    children: [run(text, { size: 24, bold: true, color: C_BLACK })],
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { before: 40, after: 40, line: 300 },
    children: [run(text)],
  });
}

function bulletRich(parts) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { before: 40, after: 40, line: 300 },
    children: parts,
  });
}

function numbered(text) {
  return new Paragraph({
    numbering: { reference: 'numbers', level: 0 },
    spacing: { before: 40, after: 40, line: 300 },
    children: [run(text)],
  });
}

function callout(text) {
  return P([run(text, { size: 20 })], {
    fill: C_CALLFILL, before: 140, after: 140, indent: { left: 140, right: 140 }, line: 320,
  });
}

/* 指标：数字第一排，说明第二排 */
function metric(num, unit, label, gold) {
  return [
    P([run(num, { size: 32, bold: true, color: gold ? C_GOLD : C_BLUE }),
       run(unit ? ' ' + unit : '', { size: 20, color: C_MUTED })],
      { before: 140, after: 0 }),
    P([run(label, { size: 19, color: C_MUTED })], { before: 0, after: 120, line: 280 }),
  ];
}

/* ---------- 图片 ---------- */
function pngSize(file) {
  const buf = fs.readFileSync(file);
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

function figure(file, caption) {
  const full = path.join(FIG, file);
  const { w, h } = pngSize(full);
  const targetW = 590;
  let targetH = Math.round(h * targetW / w);
  if (targetH > 780) { targetH = 780; }
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 180, after: 60 },
      children: [new ImageRun({
        type: 'png',
        data: fs.readFileSync(full),
        transformation: { width: targetW, height: targetH },
        altText: { title: caption, description: caption, name: caption },
      })],
    }),
    P([run(caption, { size: 18, color: C_MUTED })], { align: AlignmentType.CENTER, before: 0, after: 200, line: 280 }),
  ];
}

/* ---------- 表格 ---------- */
function cell(spec, width) {
  if (typeof spec === 'string') spec = { t: spec };
  const para = new Paragraph({
    alignment: spec.num ? AlignmentType.RIGHT : AlignmentType.LEFT,
    spacing: { before: 20, after: 20, line: 260 },
    children: [run(spec.t, { size: spec.size || 19, bold: !!spec.bold, color: spec.color || C_BLACK })],
  });
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: spec.fill || 'FFFFFF', type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [para],
  });
}

function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const headRow = new TableRow({
    tableHeader: true,
    cantSplit: true,
    children: headers.map((h, i) => cell({ t: h, bold: true, color: C_BLUE, fill: C_HEADFILL, num: i > 0 }, widths[i])),
  });
  const bodyRows = rows.map((r) => new TableRow({
    cantSplit: true,
    children: r.map((c, i) => cell(c, widths[i])),
  }));
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [headRow].concat(bodyRows),
  });
}

function gap() { return P([run('', { size: 12 })], { before: 60, after: 60 }); }

/* ================= 正文内容 ================= */
const children = [];

/* ---- 封面 ---- */
children.push(P([run('机器学习 · 蛋白质工程', { size: 19, bold: true, color: C_GOLD })], { before: 0, after: 100 }));
children.push(P([run('UGT91C1 糖基转移酶突变体活性预测模型', { size: 36, bold: true })], { before: 0, after: 120, line: 340 }));
children.push(P([run('面向候选突变体筛选的专用排序模型：数据协议重建、特征方案升级、按位点留出评测与实验数据验证', { size: 23, color: C_MUTED })], { before: 0, after: 200, line: 340 }));
children.push(P([run('目标蛋白：UGT91C1（462 aa）　|　评测口径：排序能力（Spearman / AUC / top-k）　|　报告日期：2026-10-16', { size: 19, color: C_MUTED })], { before: 0, after: 240 }));

children.push(H2('关键指标'));
metric('67', '条', '实验突变体数据 — 批次 A（Sheet1，19 条）与批次 B（F208M 组合库，48 条）合并，按批内 WT 归一化')
  .forEach((p) => children.push(p));
metric('0.477', '', '零样本进化分数 macro Spearman — 在 27 个位点留出评测下的最佳排序能力，优于线上配方 0.287')
  .forEach((p) => children.push(p));
metric('5 / 5', '', 'Sheet1 真实高活性突变体覆盖率 — 前 10 名候选即覆盖全部 5 个实测 top5 突变体', true)
  .forEach((p) => children.push(p));
metric('1.76', '倍', '批次间原始活性尺度漂移 — 直接混用两批数据会引入系统性偏差，必须做批内归一化')
  .forEach((p) => children.push(p));

children.push(callout('一句话结论：在“挑出活性最高的前 N 个候选”这一目标下，参数无关的进化零样本分数是目前最稳的排序器；在 1280 维 ESM 表征上训练的浅头受样本量限制，外推到新位点时反而退化。模型可用，但必须只按名次取前 N，不要用预测绝对值做阈值筛选。'));
children.push(new Paragraph({ children: [new PageBreak()] }));

/* ---- 01 ---- */
children.push(H1('01　项目任务与核心命题'));
children.push(T('本项目要解决的是一个明确的工程问题：在 UGT91C1 这个单一酶上，从大量待测候选突变体中，用机器学习挑出活性最高的少数几个，减少湿实验的工作量与消耗。', { size: 22, color: C_MUTED }));
children.push(H2('任务定义'));
children.push(bulletRich([run('输入：', { bold: true }), run('候选突变体的氨基酸序列（相对于野生型 WT 的突变位点与取代类型）。')]));
children.push(bulletRich([run('输出：', { bold: true }), run('候选突变体的活性排序，用于取前 N 个送去做实验验证。')]));
children.push(bulletRich([run('标签：', { bold: true }), run('相对野生型的比酶活 activity_rel（WT = 1.0）。')]));
children.push(bulletRich([run('核心指标：', { bold: true }), run('排序能力 — Spearman 相关系数、AUC（判别活性高于 WT 的能力）、以及取前 k 个候选时命中真实高活性突变体的比例。')]));
children.push(H2('为什么是排序，而不是拟合精度'));
children.push(T('这是本项目最关键的一次认知校准。活性预测的绝对数值受实验条件、底物浓度、检测批次影响极大，回归 R² 与 RMSE 会被这些系统性因素主导；而实际筛选流程只关心“哪些排前面”。因此：'));
children.push(bullet('不追求 R² 高，只看名次是否可靠；'));
children.push(bullet('不使用预测绝对值做阈值筛选（例如“预测活性 > 2 就送测”），因为绝对值的偏移会让阈值失效；'));
children.push(bullet('评测时优先看 Spearman 与 top-k 命中率，RMSE 仅作参考。'));
children.push(callout('红线（本次执行中严格守住的四件事）：① 不冻结全量微调 650M 参数（数据量仅 67 条，必然过拟合）；② 不用随机折交叉验证选超参（会虚高泛化能力）；③ 不用预测绝对值做阈值筛选；④ 不跨模型直接比较分数绝对值。'));
children.push(H2('本报告的交付范围'));
children.push(bullet('重建统一训练集与评测协议（Tier 0）；'));
children.push(bullet('新增三类冻结特征：全序列 mean-pool、突变位点窗口、masked-marginal 零样本进化分数（Tier 1）；'));
children.push(bullet('按位点留出交叉验证，对比线上配方与五组新方案的排序能力（Tier 2）；'));
children.push(bullet('用实验室真实数据（Sheet1）做样本外验证，并对候选库输出可用排名（Tier 3）。'));

/* ---- 02 ---- */
children.push(H1('02　数据重建与评测协议'));
children.push(T('数据层面的三个动作：核验标签来源、消除批次偏差、冻结一套能反映真实泛化场景的评测协议。', { size: 22, color: C_MUTED }));
children.push(H2('2.1　两批实验数据的尺度差异'));
children.push(T('实验数据来自两个独立批次，其原始比酶活不在同一尺度上：批次 A 的批内 WT 为 0.021，批次 B 的批内 WT 为 3.700，两者相差 1.76 倍。若不处理直接混用，模型会学到“批次”这个与生物活性无关的伪特征。'));
figure('fig01_data_overview.png', '图 1　两批次原始活性尺度差异（左）与数据规模（右）。批次 A 与批次 B 的活性分布在对数刻度上整体错开。')
  .forEach((x) => children.push(x));
children.push(H2('2.2　批内 WT 归一化'));
children.push(T('解决办法是各批次分别以批内 WT 为基准归一化：activity_rel = 该批实测值 / 该批 WT 实测值。归一化后两批分布回到同一尺度，WT 恒为 1.0，可以直接合并建模。'));
figure('fig02_normalization.png', '图 2　归一化前（左）两批次明显分离；归一化后（右）分布对齐，可直接合并。')
  .forEach((x) => children.push(x));
children.push(H2('2.3　突变位点分布'));
figure('fig03_mutation_map.png', '图 3　实验突变位点在 462 aa 序列上的分布。批次 A 覆盖 15 个位点，批次 B 覆盖 15 个位点，其中 208 位（F208M）出现在几乎所有批次 B 变体中，属背景位点，建模与评测时单独处理。')
  .forEach((x) => children.push(x));
children.push(H2('2.4　按位点留出交叉验证（关键协议）'));
children.push(T('评测协议决定了结论是否可信。本项目冻结以下协议：'));
children.push(P([run('Leave-one-site-out CV：', { bold: true }),
  run('对每一个非背景突变位点 p，把“所有在 p 位发生突变的变体”整体作为测试集，其余变体作训练集。一个变体若携带多个突变位点，会在多个折中被预测，取其平均作为样本外预测。')], { fill: C_CALLFILL, before: 120, after: 120, indent: { left: 140, right: 140 } }));
children.push(T('这样做评测的是「模型能否外推到从未见过的突变位点」——这正是“从候选库挑新突变体”的真实场景。随机折会把同一位点的样本同时放进训练与测试，导致分数虚高，因此被明确禁止。'));
children.push(T('最终评测集为 27 个非背景位点 / 64 个变体；背景位点 208 从评测中剔除，避免“所有变体都带同一个突变”造成的常数效应。'));
children.push(H2('2.5　数据规模与突变数分布'));
figure('fig04_nmut_vs_activity.png', '图 4　批次 B 中突变数与活性的关系：携带突变越多，越难维持高活性，单突变体的活性中位数明显更高。')
  .forEach((x) => children.push(x));
figure('fig05_site_substitution_heatmap.png', '图 5　批次 B 各位点的取代–活性图谱。空白格表示该取代组合未测。可看出同一取代类型在不同位点上的活性差异极大，位点特异性强。')
  .forEach((x) => children.push(x));

/* ---- 03 ---- */
children.push(H1('03　线上基线模型的复现与缺陷定位'));
children.push(T('在升级方案之前，先确认现有线上模型是否还能正常使用。结论是：模型文件本身已不可用，必须重训。', { size: 22, color: C_MUTED }));
children.push(H2('3.1　缺陷：跨版本加载丢截距'));
children.push(callout('问题定位：线上模型 models/xgb_esm.pkl 由旧版 XGBoost 序列化。在当前环境（XGBoost 3.0.5）下加载时会丢失 base_score 截距，导致所有预测值整体偏移约 1.4：实测 activity_rel 均值分别为 1.05 / 1.52，而直接加载预测的均值仅 -0.34 / 0.12。'));
children.push(T('影响面：平台的预测绝对数值、首页展示的 RMSE 与“准确度”指标均不可信。但排序不受影响——整体平移不改变名次，所以“按名次取前 N”的用法依然有效。为彻底消除隐患，本次在本地以完全相同配方重训模型。'));
children.push(H2('3.2　基线配方'));
children.push(P([run('特征：', { bold: true }), run('ESM2-t33-650M 全序列 mean-pool 表征（1280 维）+ 23 维领域先验（进化 PSSM、共进化、结构几何、口袋性质、稳定性 ΔΔG）　'),
  run('模型：', { bold: true }), run('XGBoost 回归（600 棵树，max_depth 4，lr 0.05）　'),
  run('目标：', { bold: true }), run('批内 WT 归一的 activity_rel。')], { fill: C_CALLFILL, before: 120, after: 120, indent: { left: 140, right: 140 } }));
children.push(H2('3.3　基线在按位点留出下的表现'));
figure('fig07_oof_scatter.png', '图 6　线上配方的样本外预测 vs 实测。整体呈正向趋势，说明基线具备基本的排序能力，但点位分散度大，绝对数值不可依赖。')
  .forEach((x) => children.push(x));
children.push(T('基线在 27 个位点留出评测下取得 macro Spearman 0.287、AUC 0.641。这个水平说明：模型能捕捉一部分信号，但外推到新位点时能力明显衰减——这正是本次方案要解决的问题。'));

/* ---- 04 ---- */
children.push(H1('04　三类冻结特征的构建'));
children.push(T('在不微调 650M 参数的前提下，提升表征对“突变”这一事件的敏感度。共构造三类特征，全部为冻结特征。', { size: 22, color: C_MUTED }));
children.push(H2('4.1　全序列 mean-pool（基线表征）'));
children.push(T('对整条突变后序列的残基表征取均值（1280 维）。优点是信息全局、稳定；缺点是单点突变被 462 个残基平均稀释，信号强度低，这也是基线外推能力弱的根本原因。'));
children.push(H2('4.2　突变位点窗口（site-window）'));
children.push(T('仅在突变位点 ± 2 个残基的邻域内取表征并平均，聚焦局部环境。同时输出剔除背景位点后的版本（site-local），消除所有变体共有的常数项。设计意图是让突变信号不被稀释。'));
children.push(H2('4.3　masked-marginal 零样本进化分数（关键新增）'));
children.push(T('这是本次最有价值的特征，思路源自 ESM-1v 的零样本突变效应打分：'));
children.push(P([run('以野生型序列为模板，把待评估位点替换为 <mask>，让语言模型给出该位点的氨基酸概率分布，然后计算对数几率：score = log p(突变氨基酸) - log p(野生型氨基酸)', { size: 20, color: C_BLUE })],
  { fill: C_CALLFILL, before: 120, after: 120, indent: { left: 140, right: 140 } }));
children.push(T('该分数衡量的是“进化过程是否接受这个取代”：分数越高，说明该取代在自然同源序列中越常见，通常越可能保持功能。它完全不依赖任何活性标签，因此不存在小样本过拟合问题，天然适合外推到新位点。本项目同时输出位点分数之和、均值、最小值和仅非背景位点的版本。'));
children.push(H2('4.4　方案对比设计'));
children.push(T('共评测 6 个方案，覆盖基线、单特征、特征组合与不同浅头：'));
children.push(table(
  ['方案', '特征', '模型头', '特征维度'],
  [
    [{ t: 'ZS', bold: true }, '零样本分数（仅非背景位点）', '无（参数无关）', { t: '1', num: true }],
    [{ t: 'M0（线上配方）', bold: true }, 'mean-pool + 先验', 'XGBoost', { t: '1303', num: true }],
    [{ t: 'M1', bold: true }, '零样本 + 先验', 'XGBoost', { t: '30', num: true }],
    [{ t: 'M2', bold: true }, '位点窗口 + 零样本 + 先验', '标准化 + PCA(32) + Ridge', { t: '1310', num: true }],
    [{ t: 'M3', bold: true }, '位点窗口 + 零样本 + 先验', 'XGBoost', { t: '1310', num: true }],
    [{ t: 'M4', bold: true }, 'mean-pool + 位点窗口 + 零样本 + 先验', '标准化 + PCA(32) + Ridge', { t: '2590', num: true }],
  ],
  [2900, 3400, 1800, 926]
));
children.push(gap());
children.push(callout('为控制过拟合做的设计约束：所有方案一律冻结 650M 主干；高维特征必须先降维再送入线性头（M2/M4 用 PCA 压到 32 维，可训练参数约 1K 量级）；不冻结主干、不微调大参数矩阵。'));

/* ---- 05 ---- */
children.push(H1('05　按位点留出评测结果'));
children.push(T('27 个位点留出、64 个变体评测。结论明确：零样本分数胜出，位点窗口表征是负贡献。', { size: 22, color: C_MUTED }));
figure('fig06_model_comparison.png', '图 7　方案对比。A 图为 Spearman（粗条为 macro、细条为 pooled），B 图为 AUC 与 Recall@10。零样本分数在两项排序指标上均最优；位点窗口方案（M2/M3）出现负相关。')
  .forEach((x) => children.push(x));
children.push(H2('5.1　完整指标表'));
children.push(table(
  ['方案', 'macro Spearman', 'pooled Spearman', 'macro AUC', 'macro Recall@10'],
  [
    [{ t: 'ZS 零样本分数', bold: true }, { t: '0.477', num: true, bold: true, color: C_BLUE }, { t: '0.401', num: true, bold: true, color: C_BLUE }, { t: '0.742', num: true, bold: true, color: C_BLUE }, { t: '0.450', num: true, bold: true, color: C_BLUE }],
    ['M0 mean-pool+先验 XGB（线上配方）', { t: '0.287', num: true }, { t: '0.281', num: true }, { t: '0.641', num: true }, { t: '0.400', num: true }],
    ['M1 零样本+先验 XGB', { t: '0.170', num: true }, { t: '0.214', num: true }, { t: '0.695', num: true }, { t: '0.300', num: true }],
    ['M4 全特征融合 PCA+Ridge', { t: '0.129', num: true }, { t: '0.210', num: true }, { t: '0.630', num: true }, { t: '0.400', num: true }],
    ['M2 位点窗口+零样本 PCA+Ridge', { t: '-0.123', num: true, color: C_MUTED }, { t: '-0.146', num: true, color: C_MUTED }, { t: '0.463', num: true }, { t: '0.450', num: true }],
    ['M3 位点窗口+零样本 XGB', { t: '-0.170', num: true, color: C_MUTED }, { t: '-0.142', num: true, color: C_MUTED }, { t: '0.446', num: true }, { t: '0.350', num: true }],
  ],
  [3400, 1400, 1400, 1400, 1426]
));
children.push(gap());
children.push(H2('5.2　三个关键结论'));
metric('0.477', 'vs 0.287', '零样本分数 macro Spearman 高于线上配方 66%，且该分数不需要任何训练、不消耗标签')
  .forEach((x) => children.push(x));
metric('-0.170', '', '位点窗口方案（M3）的 macro Spearman 为负 — 在 67 条数据规模下，位点局部表征是负贡献，最初的 Tier 1 假设不成立')
  .forEach((x) => children.push(x));
metric('0.170', 'vs 0.477', '把零样本分数交给 XGBoost 训练（M1）后，排序能力反而从 0.477 掉到 0.170 — 训练头在此数据量下是拖累而非增益', true)
  .forEach((x) => children.push(x));
children.push(H2('5.3　分批次诊断'));
figure('fig08_zeroshot_vs_activity.png', '图 8　左：零样本分数 vs 实测活性（按批次着色）。批次 A（单点突变）呈明显正相关；批次 B（组合库）相关性弱。右：各方法宏观排序能力汇总。')
  .forEach((x) => children.push(x));
children.push(table(
  ['数据子集', '构成', '零样本分数 Spearman', '说明'],
  [
    [{ t: '批次 A', bold: true }, '18 个单点突变', { t: '0.72', num: true, bold: true, color: C_BLUE }, '零样本分数表现优秀，AUC 达 0.89'],
    [{ t: '批次 B', bold: true }, '46 个 F208M 组合变体', { t: '0.23', num: true, color: C_MUTED }, '组合突变叠加是当前唯一短板'],
  ],
  [1400, 2000, 1626, 4000]
));
children.push(gap());
children.push(callout('短板定位结论：单点突变的外推能力已经可用（Spearman 0.72）；组合突变（多突变叠加）的活性预测是本项目唯一未解决的薄弱环节。后续主动学习应专门补这一类样本。'));

/* ---- 06 ---- */
children.push(H1('06　实验室实测数据（Sheet1）验证'));
children.push(T('用批次 A 的 18 个真实实测单点突变体做样本外验证。模型训练集只使用批次 B 的 48 条数据，不含 Sheet1，因此这是严格样本外预测，可直接与实验真值对比作图。', { size: 22, color: C_MUTED }));
children.push(H2('6.1　预测与实测对比'));
figure('fig12_topk_curve.png', '图 9　左：取前 k 个候选时的命中率曲线；右：高活性突变命中情况，深色块表示命中真实 top5。')
  .forEach((x) => children.push(x));
figure('fig09_candidate_score_dist.png', '图 10　候选库（1371 条）的零样本分数分布与突变数构成，作为 6.3 节应用部分的数据背景。')
  .forEach((x) => children.push(x));
children.push(H2('6.2　逐条预测明细'));
children.push(T('下表为 18 个真实突变体的完整预测结果（WT 作为标定点单列，不参与突变体排名）。'));
children.push(table(
  ['突变体', '实测 activity_rel', 'ESM+XGB 预测', '零样本分数', '实测名次', '预测名次', '零样本名次'],
  [
    [{ t: 'F379L', bold: true }, { t: '4.19', num: true }, { t: '1.487', num: true }, { t: '0.267', num: true }, { t: '1', num: true }, { t: '3', num: true, bold: true, color: C_BLUE }, { t: '4', num: true, bold: true, color: C_BLUE }],
    [{ t: 'F379E', bold: true }, { t: '3.29', num: true }, { t: '0.972', num: true }, { t: '-3.389', num: true }, { t: '2', num: true }, { t: '10', num: true, bold: true, color: C_BLUE }, { t: '10', num: true, bold: true, color: C_BLUE }],
    [{ t: 'C121S', bold: true }, { t: '2.43', num: true }, { t: '1.159', num: true }, { t: '2.307', num: true }, { t: '3', num: true }, { t: '6', num: true, bold: true, color: C_BLUE }, { t: '1', num: true, bold: true, color: C_BLUE }],
    [{ t: 'V129G', bold: true }, { t: '2.38', num: true }, { t: '2.138', num: true }, { t: '-1.178', num: true }, { t: '4', num: true }, { t: '1', num: true, bold: true, color: C_BLUE }, { t: '6', num: true, bold: true, color: C_BLUE }],
    [{ t: 'C445S', bold: true }, { t: '2.00', num: true }, { t: '1.919', num: true }, { t: '1.347', num: true }, { t: '5', num: true }, { t: '2', num: true, bold: true, color: C_BLUE }, { t: '3', num: true, bold: true, color: C_BLUE }],
    ['WT（标定）', { t: '1.00', num: true }, { t: '1.000', num: true }, { t: '0.000', num: true }, { t: '—', num: true }, { t: '—', num: true }, { t: '—', num: true }],
    ['F379W', { t: '0.76', num: true }, { t: '0.783', num: true }, { t: '-1.818', num: true }, { t: '7', num: true }, { t: '14', num: true }, { t: '8', num: true }],
    ['F186E', { t: '0.71', num: true }, { t: '1.334', num: true }, { t: '1.976', num: true }, { t: '8', num: true }, { t: '4', num: true }, { t: '2', num: true }],
    ['V127L', { t: '0.62', num: true }, { t: '0.710', num: true }, { t: '-1.795', num: true }, { t: '9', num: true }, { t: '16', num: true }, { t: '7', num: true }],
    ['F369E', { t: '0.57', num: true }, { t: '1.127', num: true }, { t: '-6.114', num: true }, { t: '10', num: true }, { t: '8', num: true }, { t: '13', num: true }],
    ['F70E', { t: '0.29', num: true }, { t: '1.172', num: true }, { t: '-7.287', num: true }, { t: '11', num: true }, { t: '5', num: true }, { t: '15', num: true }],
    ['F453E', { t: '0.29', num: true }, { t: '0.886', num: true }, { t: '-11.916', num: true }, { t: '11', num: true }, { t: '11', num: true }, { t: '19', num: true }],
    ['V78G', { t: '0.24', num: true }, { t: '0.735', num: true }, { t: '-6.365', num: true }, { t: '13', num: true }, { t: '15', num: true }, { t: '14', num: true }],
    ['V278G', { t: '0.19', num: true }, { t: '0.532', num: true }, { t: '-7.671', num: true }, { t: '14', num: true }, { t: '19', num: true }, { t: '16', num: true }],
    ['V127G', { t: '0.19', num: true }, { t: '0.587', num: true }, { t: '-3.367', num: true }, { t: '14', num: true }, { t: '17', num: true }, { t: '9', num: true }],
    ['F106D', { t: '0.19', num: true }, { t: '1.141', num: true }, { t: '-3.483', num: true }, { t: '14', num: true }, { t: '7', num: true }, { t: '11', num: true }],
    ['F226E', { t: '0.19', num: true }, { t: '0.835', num: true }, { t: '-4.590', num: true }, { t: '14', num: true }, { t: '12', num: true }, { t: '12', num: true }],
    ['Y277R', { t: '0.19', num: true }, { t: '0.810', num: true }, { t: '-8.378', num: true }, { t: '14', num: true }, { t: '13', num: true }, { t: '17', num: true }],
    ['V143G', { t: '0.14', num: true }, { t: '0.562', num: true }, { t: '-8.915', num: true }, { t: '19', num: true }, { t: '18', num: true }, { t: '18', num: true }],
  ],
  [1426, 1300, 1300, 1300, 1200, 1200, 1300]
));
children.push(gap());
children.push(H2('6.3　验证结论'));
metric('0.664', '', 'ESM+XGB 配方在 18 个真实突变体上的 Spearman — 样本外仍保持正向排序能力')
  .forEach((x) => children.push(x));
metric('0.721', '', '零样本分数的 Spearman — 不训练、不用标签，排序能力仍高于训练模型', true)
  .forEach((x) => children.push(x));
metric('5 / 5', '', '两种方法各自的前 10 名候选均覆盖全部 5 个真实高活性突变体（F379L、F379E、C121S、V129G、C445S）')
  .forEach((x) => children.push(x));
children.push(callout('可直接写进报告的核心证据：模型在严格样本外条件下，用前 10 个候选就覆盖了全部 5 个实验验证的高活性突变体。这支撑“模型可用于候选筛选”的结论。注意预测绝对值被系统性压缩（实测 4.19 的 F379L 仅预测 1.49），因此请只用名次，不用绝对值。'));

/* ---- 07 ---- */
children.push(H1('07　候选突变体库排序'));
children.push(T('把最优排序器应用到候选库：1371 条候选，按零样本分数（剔除背景位点后）降序排列，输出可直接使用的排名表。', { size: 22, color: C_MUTED }));
figure('fig10_candidate_top20.png', '图 11　候选库零样本排名前 20。分数为各突变位点对数几率之和。')
  .forEach((x) => children.push(x));
figure('fig11_nmut_vs_zeroshot.png', '图 12　零样本分数的两种聚合方式。左：求和会随突变数增加而累加负分，天然偏向单突变体；右：按位点取平均可消除这一数量偏差。做组合库筛选时建议两列同时参考。')
  .forEach((x) => children.push(x));
children.push(callout('使用提醒：zs_sum 对多突变体不公平（突变越多、负分累加越多），因此排名表前列几乎全是单突变体。若目标是筛选多突变组合，请改用 zs_nonbg_mean 列排序，或对 zs_sum 做突变数归一化。'));
children.push(H2('输出文件'));
children.push(table(
  ['文件', '内容', '行数'],
  [
    [{ t: 'results/zeroshot_candidate_ranking.csv', size: 18 }, '候选库按零样本分数降序排名，含 zs_sum / zs_nonbg_sum / zs_min / zs_mean / zs_nonbg_mean 多列', { t: '1371', num: true }],
    [{ t: 'results/sheet1_predictions.csv', size: 18 }, 'Sheet1 实验突变体的样本外预测与三套名次', { t: '19', num: true }],
    [{ t: 'results/tier_cv_metrics.csv', size: 18 }, '六组方案的按位点留出评测指标汇总', { t: '6', num: true }],
    [{ t: 'results/tier_oof_predictions.csv', size: 18 }, '各方案的样本外（OOF）预测明细', { t: '64', num: true }],
  ],
  [4000, 3500, 1526]
));

/* ---- 08 ---- */
children.push(H1('08　结论、局限与下一步'));
children.push(H2('8.1　主要结论'));
children.push(numbered('数据协议是前提。两批次数据存在 1.76 倍尺度漂移，必须按批内 WT 归一化后合并；评测必须用按位点留出，随机折会虚高泛化能力。'));
children.push(numbered('零样本进化分数是最稳的排序信号。在 27 个位点留出评测下 macro Spearman 0.477、AUC 0.742，优于线上配方（0.287 / 0.641），且不消耗任何标签。'));
children.push(numbered('位点窗口表征在当前数据量下无效甚至有害。M2/M3 的 Spearman 为负，原“聚焦局部环境”的假设未获支持。'));
children.push(numbered('小样本下加训练头会损伤排序能力。把零样本分数交给 XGBoost 训练后（M1），表现从 0.477 掉到 0.170。'));
children.push(numbered('模型可用于候选筛选。在 Sheet1 的 18 个真实突变体上，前 10 名候选覆盖全部 5 个高活性突变体。'));
children.push(numbered('线上模型文件必须重训。旧版序列化的 xgb_esm.pkl 在当前 XGBoost 下丢失截距，预测整体偏移约 1.4，平台显示的精度指标不可信。'));
children.push(H2('8.2　明确局限'));
children.push(bulletRich([run('组合突变预测是短板：', { bold: true }), run('批次 B（46 个 F208M 组合变体）的零样本 Spearman 仅 0.23。')]));
children.push(bulletRich([run('预测绝对值不可用：', { bold: true }), run('实测 4.19 的突变体预测仅 1.49，存在系统性压缩，且批次标定差异尚未完全消除。')]));
children.push(bulletRich([run('样本量小：', { bold: true }), run('67 条数据下，任何在 1280 维表征上训练的模型都难以稳定收敛，这也是零样本方案胜出的根本原因。')]));
children.push(H2('8.3　后续建议（按优先级）'));
metric('P0', '', '用本地重训的同配方模型替换线上 xgb_esm.pkl，修复平台首页的 RMSE / 准确度显示，并明确标注“仅按名次使用”')
  .forEach((x) => children.push(x));
metric('P1', '', '主动学习回灌，专门补组合突变样本（多突变叠加），这是唯一被验证的短板；每轮实验后把新数据按批内 WT 归一化后并入训练集')
  .forEach((x) => children.push(x));
metric('P2', '', '把零样本分数作为候选筛选的第一道排序器稳定上线；放弃位点窗口特征方向，不再投入')
  .forEach((x) => children.push(x));
metric('P3', '', '数据量积累到 200 条以上后，再考虑轻量微调（LoRA r=4 仅挂最后两层）或适配器方案；在此之前不投入')
  .forEach((x) => children.push(x));
children.push(callout('一句话给决策者：当前阶段不必追求“更准的绝对预测”，而应追求“更稳的名次”。零样本分数已经提供了这一能力，把资源投到补齐组合突变实验数据上，收益远大于继续在模型结构上做文章。'));

/* ---- 09 ---- */
children.push(H1('09　脚本清单与复现步骤'));
children.push(T('全部脚本为新增文件，不改动既有线上代码。按顺序执行即可复现本报告全部结论。', { size: 22, color: C_MUTED }));
children.push(table(
  ['步骤', '脚本', '作用', '产出'],
  [
    [{ t: '1', num: true }, { t: 'scripts/build_dataset.py', size: 18 }, '合并两批次、批内 WT 归一化、自动识别背景位点', { t: 'data/train_dataset.csv', size: 18 }],
    [{ t: '2', num: true }, { t: 'scripts/extract_features_v2.py', size: 18 }, '抽 ESM2-650M 的 mean-pool / 位点窗口 / 零样本分数', { t: 'data/feat_*.npy', size: 18 }],
    [{ t: '3', num: true }, { t: 'scripts/train_eval_v2.py', size: 18 }, '按位点留出 CV，对比六组方案', { t: 'results/tier_cv_metrics.csv', size: 18 }],
    [{ t: '4', num: true }, { t: 'scripts/predict_sheet1.py', size: 18 }, '用线上同配方重训并对 Sheet1 做样本外预测', { t: 'results/sheet1_predictions.csv', size: 18 }],
    [{ t: '5', num: true }, { t: 'scripts/score_candidates_zeroshot.py', size: 18 }, '用零样本分数对候选库排序', { t: 'results/zeroshot_candidate_ranking.csv', size: 18 }],
    [{ t: '6', num: true }, { t: 'scripts/plot_sheet1_comparison.py', size: 18 }, 'Sheet1 预测对比图', { t: 'PNG', size: 18 }],
    [{ t: '7', num: true }, { t: 'scripts/make_report_figures.py', size: 18 }, '本报告全部 12 张图', { t: 'results/figures/*.png', size: 18 }],
  ],
  [800, 2900, 3800, 1526]
));
children.push(gap());
children.push(H2('执行命令'));
children.push(P([
  run('python scripts/build_dataset.py', { size: 19, color: C_BLUE }),
], { fill: C_CALLFILL, before: 100, after: 0, indent: { left: 140 } }));
['extract_features_v2.py', 'train_eval_v2.py', 'predict_sheet1.py', 'score_candidates_zeroshot.py', 'make_report_figures.py'].forEach((s, i, arr) => {
  children.push(P([run('python scripts/' + s, { size: 19, color: C_BLUE })],
    { fill: C_CALLFILL, before: 0, after: i === arr.length - 1 ? 160 : 0, indent: { left: 140 } }));
});
children.push(H2('运行环境'));
children.push(bullet('Python 3.13 · PyTorch 2.9（CUDA）· esm 预训练权重已本地缓存（esm2_t33_650M_UR50D）'));
children.push(bullet('XGBoost 3.0.5 · scikit-learn · pandas · numpy · matplotlib'));
children.push(bullet('注意：一次只运行一个模型任务，避免算力争抢'));

/* ================= 组装 ================= */
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: FONT, size: 21 } },
    },
    paragraphStyles: [
      {
        id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: C_BLACK },
        paragraph: { spacing: { before: 300, after: 140 }, outlineLevel: 0, keepNext: false, keepLines: false },
      },
      {
        id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: C_BLACK },
        paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1, keepNext: false, keepLines: false },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: 'bullets',
        levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 640, hanging: 320 } } } }],
      },
      {
        reference: 'numbers',
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 640, hanging: 320 } } } }],
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    children: children,
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(OUT, buffer);
  console.log('Saved:', OUT, (buffer.length / 1024).toFixed(0) + ' KB');
});
