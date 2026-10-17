/* UGT91C1 项目报告 — 分页导航 */
(function () {
  var pages = Array.prototype.slice.call(document.querySelectorAll('.page'));
  var nav = document.getElementById('nav');

  function shortTitle(t) {
    var map = {
      '封面': '封面',
      '任务与背景': '任务定义',
      '数据与协议': '数据与协议',
      '基线诊断': '基线诊断',
      '特征方案': '特征方案',
      '模型对比结果': '评测结果',
      '实验数据验证': '实验验证',
      '候选库应用': '候选库排序',
      '结论与建议': '结论建议',
      '复现说明': '复现说明'
    };
    return map[t] || t;
  }

  // 建立侧栏导航与页脚翻页控件
  pages.forEach(function (page, i) {
    var label = shortTitle(page.dataset.title || ('第 ' + (i + 1) + ' 节'));
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.innerHTML = '<span class="no">' + String(i + 1).padStart(2, '0') + '</span>' + label;
    btn.addEventListener('click', function () { go(i); });
    nav.appendChild(btn);

    var pager = page.querySelector('.pager');
    if (pager) {
      var prev = document.createElement('button');
      prev.type = 'button';
      prev.textContent = '← 上一节';
      prev.disabled = i === 0;
      prev.addEventListener('click', function () { go(i - 1); });

      var pos = document.createElement('span');
      pos.className = 'pos';
      pos.textContent = String(i + 1) + ' / ' + pages.length;

      var next = document.createElement('button');
      next.type = 'button';
      next.textContent = '下一节 →';
      next.disabled = i === pages.length - 1;
      next.addEventListener('click', function () { go(i + 1); });

      pager.appendChild(prev);
      pager.appendChild(pos);
      pager.appendChild(next);
    }
  });

  function go(index) {
    if (index < 0 || index >= pages.length) return;
    pages.forEach(function (p, i) { p.classList.toggle('active', i === index); });
    Array.prototype.forEach.call(nav.children, function (b, i) {
      b.classList.toggle('active', i === index);
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // 键盘左右方向键翻页
  document.addEventListener('keydown', function (e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    var current = pages.findIndex(function (p) { return p.classList.contains('active'); });
    if (e.key === 'ArrowRight') go(current + 1);
  else if (e.key === 'ArrowLeft') go(current - 1);
  });

  go(0);
})();
