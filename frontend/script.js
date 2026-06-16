const $ = (id) => document.getElementById(id);

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  return await res.json();
}

function fmtTime(ms) {
  const d = new Date(ms);
  return d.toLocaleTimeString();
}

function updateMetrics(m) {
  $('totalPeople').textContent = m.total_people ?? 0;
  $('queueLen').textContent = m.active_queue_length ?? 0;
  $('ewt').textContent = `${m.estimated_wait_time_sec ?? 0}s`;
  if ($('throughput')) {
    const tp = m.throughput_per_min != null ? m.throughput_per_min : 0;
    $('throughput').textContent = `${tp}/min`;
  }
  const alert = m.alert ? 'ALERT' : 'OK';
  $('alertStatus').textContent = alert;
  const badge = $('alertBadge');
  badge.textContent = alert;
  badge.style.background = m.alert ? '#ef4444' : '#10b981';
  
  // Dynamic Queue Cards
  const container = $('queuesContainer');
  const recommendedBadge = $('recommendedQueue');
  
  if (container && m.queue_metrics) {
    container.innerHTML = ''; // Clear current
    
    // Sort queues by length (longest first)
    const sortedQueues = Object.entries(m.queue_metrics).sort((a, b) => b[1].count - a[1].count);
    
    // Find recommended queue (lowest EWT)
    let minEwt = Infinity;
    let recommended = null;
    for (const [name, q] of Object.entries(m.queue_metrics)) {
      if (q.est_wait_time < minEwt) {
        minEwt = q.est_wait_time;
        recommended = name;
      }
    }
    
    if (recommended && Object.keys(m.queue_metrics).length > 1) {
      recommendedBadge.textContent = `Recommended: ${recommended}`;
      recommendedBadge.style.display = 'block';
    } else {
      recommendedBadge.style.display = 'none';
    }
    
    for (const [name, q] of sortedQueues) {
      const card = document.createElement('div');
      const isRecommended = name === recommended && Object.keys(m.queue_metrics).length > 1;
      card.className = `queue-card ${q.alert ? 'alert' : ''} ${isRecommended ? 'recommended' : ''}`;
      
      const alertBadge = q.alert ? '<span style="color:#ef4444">⚠</span>' : '';
      const recLabel = isRecommended ? '<span class="rec-label">Recommended</span>' : '';
      
      card.innerHTML = `
        <h3>${name} ${alertBadge}</h3>
        ${recLabel}
        <div class="queue-stat">
          <span>Length</span>
          <span>${q.count}</span>
        </div>
        <div class="queue-stat">
          <span>Avg Service</span>
          <span>${q.avg_service_time}s</span>
        </div>
        <div class="queue-stat">
          <span>Est Wait</span>
          <span>${q.est_wait_time}s</span>
        </div>
      `;
      container.appendChild(card);
    }
  } else if (container && m.per_roi) {
      // Fallback for old structure if needed (though backend updated)
      // ... (omitted for brevity as we updated backend)
  }

  if (m.session) {
    $('sessionMaxQueue').textContent = m.session.max_queue_length;
    $('sessionAvgQueue').textContent = m.session.avg_queue_length;
    $('sessionUnique').textContent = m.session.total_unique_people;
  }
}

function appendHistoryRow(m) {
  const tbody = document.querySelector('#historyTable tbody');
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${fmtTime(m.timestamp_ms)}</td>
    <td>${m.total_people}</td>
    <td>${m.active_queue_length}</td>
    <td>${m.estimated_wait_time_sec}s</td>
    <td>${m.alert ? 'Yes' : 'No'}</td>
  `;
  tbody.appendChild(tr);
}

let uploadedPath = null;
let es = null;
let frameToHistory = 0;
let queueChart = null;
let authToken = localStorage.getItem('authToken');
let notifications = [];

function updateAuthUI() {
  const overlay = $('loginOverlay');
  const logoutBtn = $('logoutBtn');
  if (authToken) {
    overlay.style.display = 'none';
    logoutBtn.style.display = 'block';
  } else {
    overlay.style.display = 'flex';
    logoutBtn.style.display = 'none';
  }
}

$('loginForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const username = $('username').value;
  const password = $('password').value;
  const err = $('loginError');
  try {
    const res = await fetch('/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    if (res.ok) {
      const data = await res.json();
      authToken = data.access_token;
      localStorage.setItem('authToken', authToken);
      updateAuthUI();
      err.style.display = 'none';
    } else {
      err.style.display = 'block';
    }
  } catch (e) {
    err.style.display = 'block';
  }
});

$('logoutBtn').addEventListener('click', () => {
  authToken = null;
  localStorage.removeItem('authToken');
  updateAuthUI();
});

function addNotification(msg) {
  notifications.unshift({ msg, time: new Date(), read: false });
  updateNotificationUI();
}

function updateNotificationUI() {
  const badge = $('notificationBadge');
  const list = $('notificationList');
  if (!badge || !list) return;
  
  const unreadCount = notifications.filter(n => !n.read).length;
  
  if (unreadCount > 0) {
    badge.textContent = unreadCount;
    badge.style.display = 'flex';
  } else {
    badge.style.display = 'none';
  }
  
  if (notifications.length === 0) {
    list.innerHTML = '<div class="notification-empty">No new notifications</div>';
    return;
  }
  
  list.innerHTML = notifications.map((n, i) => `
    <div class="notification-item ${n.read ? 'read' : 'unread'}" onclick="markRead(${i})">
      <div style="${n.read ? '' : 'font-weight:600'}">${n.msg}</div>
      <div class="time">${n.time.toLocaleTimeString()}</div>
    </div>
  `).join('');
}

window.markRead = (index) => {
  notifications[index].read = true;
  updateNotificationUI();
};

$('notificationBtn').addEventListener('click', () => {
  const dropdown = $('notificationDropdown');
  dropdown.style.display = dropdown.style.display === 'none' ? 'block' : 'none';
});

// Close dropdown on click outside
window.addEventListener('click', (e) => {
  if (!e.target.closest('.notification-container')) {
    $('notificationDropdown').style.display = 'none';
  }
});

let roiState = {
  mode: null,
  drawing: [],
  rois: {},
  canvas: null,
  ctx: null
};

function initChart() {
  const ctx = document.getElementById('queueChart').getContext('2d');
  queueChart = new Chart(ctx, {
      type: 'line',
      data: {
          labels: [],
          datasets: [{
              label: 'Active Queue Length',
              data: [],
              borderColor: 'rgb(0, 255, 255)', // Cyan to match ROI
              backgroundColor: 'rgba(0, 255, 255, 0.1)',
              tension: 0.3,
              fill: true,
              pointRadius: 0
          }]
      },
      options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          scales: {
              x: { display: false }, 
              y: { 
                  beginAtZero: true, 
                  grid: { color: '#334155' },
                  ticks: { color: '#94a3b8' }
              }
          },
          plugins: {
              legend: { display: false }
          }
      }
  });
}

function initROI() {
  roiState.canvas = $('roiCanvas');
  roiState.ctx = roiState.canvas.getContext('2d');
  const img = $('processedFrame');
  const ph = $('videoPlaceholder');
  const queueBtn = $('roiQueueBtn');
  const cashierBtn = $('roiCashierBtn');
  const clearBtn = $('roiClearBtn');
  const saveBtn = $('roiSaveBtn');
  const manualCheckbox = $('roiManualCheckbox');
  const autoCheckbox = $('roiAutoCheckbox');
  const nameInput = $('roiQueueName');
  
  const updateROIMode = async (mode) => {
    // Sync UI
    if (mode === 'manual') {
      manualCheckbox.checked = true;
      autoCheckbox.checked = false;
      // Removed clearing to preserve manual ROIs
      roiState.drawing = [];
      drawOverlay();
    } else {
      manualCheckbox.checked = false;
      autoCheckbox.checked = true;
    }
    
    await fetch('/config', {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`
      },
      body: JSON.stringify({ 
        roi_mode: mode,
        rois: mode === 'manual' ? {} : undefined
      })
    });
  };

  manualCheckbox.addEventListener('change', () => {
    if (manualCheckbox.checked) updateROIMode('manual');
    else manualCheckbox.checked = true; // Prevent unchecking if it's the active one
  });

  autoCheckbox.addEventListener('change', () => {
    if (autoCheckbox.checked) updateROIMode('auto');
    else autoCheckbox.checked = true; // Prevent unchecking if it's the active one
  });

  function resizeCanvas() {
    if (img.style.display === 'none') {
      roiState.canvas.style.display = 'none';
      return;
    }
    const container = document.querySelector('.video-container');
    const contRect = container.getBoundingClientRect();
    const imgRect = img.getBoundingClientRect();
    const width = Math.round(imgRect.width);
    const height = Math.round(imgRect.height);
    roiState.canvas.width = width;
    roiState.canvas.height = height;
    roiState.canvas.style.width = `${width}px`;
    roiState.canvas.style.height = `${height}px`;
    roiState.canvas.style.left = `${imgRect.left - contRect.left}px`;
    roiState.canvas.style.top = `${imgRect.top - contRect.top}px`;
    roiState.canvas.style.display = 'block';
    drawOverlay();
  }
  function getScale() {
    const sx = img.naturalWidth / img.clientWidth;
    const sy = img.naturalHeight / img.clientHeight;
    return { sx, sy };
  }
  function drawPoly(points, color) {
    if (!points || points.length === 0) return;
    const ctx = roiState.ctx;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.closePath();
    ctx.stroke();
    // draw points
    ctx.fillStyle = color;
    for (const [px, py] of points) {
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  function drawOverlay() {
    const ctx = roiState.ctx;
    if (!ctx) return;
    ctx.clearRect(0, 0, roiState.canvas.width, roiState.canvas.height);
    const qname = nameInput.value || 'Queue_1';
    const existing = roiState.rois[qname] || {};
    if (existing.queue_roi_display) drawPoly(existing.queue_roi_display, 'cyan');
    if (existing.cashier_roi_display) drawPoly(existing.cashier_roi_display, 'red');
    if (roiState.drawing.length > 0) {
      const color = roiState.mode === 'queue' ? 'cyan' : 'red';
      drawPoly(roiState.drawing, color);
    }
  }
  roiState.canvas.addEventListener('click', (e) => {
    if (!roiState.mode) return;
    const rect = roiState.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    roiState.drawing.push([x, y]);
    drawOverlay();
  });
  queueBtn.addEventListener('click', () => {
    roiState.mode = 'queue';
    roiState.drawing = [];
    drawOverlay();
  });
  cashierBtn.addEventListener('click', () => {
    roiState.mode = 'cashier';
    roiState.drawing = [];
    drawOverlay();
  });
  clearBtn.addEventListener('click', () => {
    roiState.drawing = [];
    drawOverlay();
  });
  saveBtn.addEventListener('click', async () => {
    const qname = nameInput.value || 'Queue_1';
    if (!roiState.rois[qname]) roiState.rois[qname] = {};
    const { sx, sy } = getScale();
    if (roiState.mode === 'queue' && roiState.drawing.length >= 3) {
      roiState.rois[qname].queue_roi = roiState.drawing.map(([x, y]) => [Math.round(x * sx), Math.round(y * sy)]);
      roiState.rois[qname].queue_roi_display = roiState.drawing.slice();
    } else if (roiState.mode === 'cashier' && roiState.drawing.length >= 3) {
      roiState.rois[qname].cashier_roi = roiState.drawing.map(([x, y]) => [Math.round(x * sx), Math.round(y * sy)]);
      roiState.rois[qname].cashier_roi_display = roiState.drawing.slice();
    }
    roiState.drawing = [];
    drawOverlay();
    await fetch('/config', { 
      method: 'POST', 
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`
      }, 
      body: JSON.stringify({ rois: mapRoisForBackend(roiState.rois) }) 
    });
  });
  function mapRoisForBackend(rois) {
    const out = {};
    for (const [name, cfg] of Object.entries(rois)) {
      out[name] = {
        queue_roi: cfg.queue_roi || [],
        cashier_roi: cfg.cashier_roi || []
      };
    }
    return out;
  }
  img.addEventListener('load', resizeCanvas);
  window.addEventListener('resize', resizeCanvas);
  // Ensure we reposition the canvas after each new frame arrives
  const origOnMessageSetter = Object.getOwnPropertyDescriptor(EventSource.prototype, 'onmessage');
  // We won't override EventSource; instead, bind a MutationObserver to update when image src changes
  const obs = new MutationObserver(() => resizeCanvas());
  obs.observe(img, { attributes: true, attributeFilter: ['src', 'style'] });
  fetchJSON('/config', { headers: { 'Authorization': `Bearer ${authToken}` } }).then(cfg => {
    const rois = (cfg && cfg.rois) ? cfg.rois : {};
    if (cfg && cfg.roi_mode) {
      if (cfg.roi_mode === 'auto') {
        autoCheckbox.checked = true;
        manualCheckbox.checked = false;
      } else {
        autoCheckbox.checked = false;
        manualCheckbox.checked = true;
      }
    }
    roiState.rois = {};
    for (const [name, c] of Object.entries(rois)) {
      roiState.rois[name] = {
        queue_roi: c.queue_roi || [],
        cashier_roi: c.cashier_roi || []
      };
    }
    resizeCanvas();
  }).catch(()=>{});
}
function bindUpload() {
  initChart();
  initROI();
  updateAuthUI();
  const uploadBtn = $('uploadBtn');
  const fileInput = $('fileInput');
  const startBtn = $('startBtn');
  const pauseBtn = $('pauseBtn');
  const resumeBtn = $('resumeBtn');
  const stopBtn = $('stopBtn');
  uploadBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', async () => {
    const file = fileInput.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch('/upload-video', { method: 'POST', body: fd });
    const data = await res.json();
    uploadedPath = data.path;
    startBtn.disabled = !uploadedPath;
    // Reset UI ROI state and clear backend rois so only manually selected queues apply
    roiState.rois = {};
    roiState.drawing = [];
    drawOverlay();
    try { 
      await fetch('/config', { 
        method: 'POST', 
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        }, 
        body: JSON.stringify({ rois: {} }) 
      }); 
    } catch {}
    // Fetch first frame preview to enable ROI drawing
    if (uploadedPath) {
      try {
        const pre = await fetch('/first-frame', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: uploadedPath }) });
        const pjson = await pre.json();
        if (pjson.frame_b64) {
          const img = $('processedFrame');
          const ph = $('videoPlaceholder');
          
          img.onload = () => {
             // Ensure canvas matches image after it actually renders
             resizeCanvas();
          };
          
          img.src = 'data:image/jpeg;base64,' + pjson.frame_b64;
          img.style.display = 'block';
          ph.style.display = 'none';
          
          // Re-enable start button
          startBtn.disabled = false;
        } else {
           console.error('Preview failed:', pjson.error);
        }
      } catch (e) {
        console.error('Failed to load first frame preview:', e);
      }
    }
  });
  startBtn.addEventListener('click', async () => {
    if (!uploadedPath) return;
    await fetch('/start-processing', { 
      method: 'POST', 
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`
      }, 
      body: JSON.stringify({ path: uploadedPath }) 
    });
    $('currentStage').textContent = 'Starting';
    $('currentFrame').textContent = '0';
    $('totalFrames').textContent = '0';
    startBtn.style.display = 'none';
    pauseBtn.style.display = 'block';
    stopBtn.style.display = 'block';
    
    if (queueChart) {
      queueChart.data.labels = [];
      queueChart.data.datasets[0].data = [];
      queueChart.update();
    }
    if (es) es.close();
    es = new EventSource('/events');
    frameToHistory = 0;
    es.onmessage = (ev) => {
      const payload = JSON.parse(ev.data);
      if (payload.type === 'status') {
        if (payload.stage) {
          $('currentStage').textContent = payload.stage;
          if (payload.stage === 'Paused') {
            pauseBtn.style.display = 'none';
            resumeBtn.style.display = 'block';
            resizeCanvas(); // Ensure canvas is ready for drawing
          } else {
            pauseBtn.style.display = 'block';
            resumeBtn.style.display = 'none';
          }
        }
        if (payload.frame_index !== undefined) $('currentFrame').textContent = payload.frame_index;
        if (payload.total_frames !== undefined) $('totalFrames').textContent = payload.total_frames;
      } else if (payload.type === 'frame') {
        $('currentStage').textContent = payload.stage;
        $('currentFrame').textContent = payload.frame_index;
        $('totalFrames').textContent = payload.total_frames;
        if (payload.frame_b64) {
          const img = $('processedFrame');
          const ph = $('videoPlaceholder');
          img.src = 'data:image/jpeg;base64,' + payload.frame_b64;
          if (img.style.display === 'none') {
            img.style.display = 'block';
            ph.style.display = 'none';
          }
        }
        if (payload.metrics) {
          updateMetrics(payload.metrics);
          
          if (queueChart) {
             queueChart.data.labels.push('');
             queueChart.data.datasets[0].data.push(payload.metrics.active_queue_length);
             if (queueChart.data.labels.length > 100) {
                 queueChart.data.labels.shift();
                 queueChart.data.datasets[0].data.shift();
             }
             queueChart.update();
          }

          frameToHistory++;
          if (frameToHistory >= 30) {
            frameToHistory = 0;
            appendHistoryRow(payload.metrics);
          }
        }
      } else if (payload.type === 'complete') {
        $('currentStage').textContent = 'Processing Complete';
        startBtn.disabled = false;
        es.close();
        if (payload.summary) {
          const s = payload.summary;
          addNotification(`Processing Complete: ${s.total_unique_people} served, Avg Wait ${s.avg_queue_length}s, Peak ${s.max_queue_length}`);
        }
      } else if (payload.type === 'error') {
        $('currentStage').textContent = 'Error: ' + payload.message;
        console.error('Processing Error:', payload.message);
        startBtn.disabled = false;
        es.close();
      }
    };
    es.onerror = (err) => {
      console.error('SSE Connection Error:', err);
      // If the connection is closed or fails, we might want to stop or show error
      if (es.readyState === EventSource.CLOSED) {
          $('currentStage').textContent = 'Connection Lost';
          startBtn.disabled = false;
      }
    };
  });

  pauseBtn.addEventListener('click', async () => {
    await fetch('/pause-processing', { 
      method: 'POST', 
      headers: { 'Authorization': `Bearer ${authToken}` } 
    });
  });

  resumeBtn.addEventListener('click', async () => {
    await fetch('/resume-processing', { 
      method: 'POST', 
      headers: { 'Authorization': `Bearer ${authToken}` } 
    });
  });

  stopBtn.addEventListener('click', async () => {
    await fetch('/stop-processing', { 
      method: 'POST', 
      headers: { 'Authorization': `Bearer ${authToken}` } 
    });
    if (es) es.close();
    $('currentStage').textContent = 'Stopped';
    startBtn.style.display = 'block';
    pauseBtn.style.display = 'none';
    resumeBtn.style.display = 'none';
    stopBtn.style.display = 'none';
  });
}

bindUpload();
