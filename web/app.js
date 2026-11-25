const state = {
  frames: 120,
  jobId: null,
  poll: null,
  report: null,
  uploadId: null,
  benchmark: null,
};

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ helpers */
function fmt(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

function setStatus(text, live = false) {
  $("engineStatus").textContent = text;
  $("pulse").classList.toggle("on", live);
}

function setHint(text, isError = false) {
  $("runHint").textContent = text;
  $("runHint").classList.toggle("error", isError);
}

/* --------------------------------------------------------------- run control */
document.querySelectorAll("#framesSeg button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("#framesSeg button").forEach((b) => b.classList.remove("active"));
    button.classList.add("active");
    state.frames = Number(button.dataset.frames);
  });
});

$("pickFile").addEventListener("click", () => $("fileInput").click());

$("fileInput").addEventListener("change", async (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  try {
    setHint(`Uploading ${file.name} (${(file.size / 1048576).toFixed(1)} MB)…`);
    const response = await fetch(`/api/upload?name=${encodeURIComponent(file.name)}`, {
      method: "POST",
      body: file,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "upload failed");
    state.uploadId = data.upload_id;
    $("demoChip").textContent = `upload · ${file.name.slice(0, 28)}`;
    setHint("Upload complete. Starting analysis…");
    startJob("upload", state.frames, data.upload_id);
  } catch (error) {
    setHint(error.message, true);
  }
});

$("runDemo").addEventListener("click", () => startJob("demo", state.frames));

async function startJob(source, frames, uploadId) {
  try {
    const button = $("runDemo");
    button.disabled = true;
    setStatus("starting engine", true);
    $("videoPlaceholder").style.display = "grid";
    $("stream").classList.remove("on");
    $("downloadLink").hidden = true;
    $("progressBar").style.width = "0%";
    setHint("Loading detector and initialising identity memory…");

    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, max_frames: frames, upload_id: uploadId }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "could not start job");

    state.jobId = data.job_id;
    state.report = null;
    pollJob();
  } catch (error) {
    setHint(error.message, true);
    setStatus("engine idle");
    $("runDemo").disabled = false;
  }
}

function pollJob() {
  clearTimeout(state.poll);
  state.poll = setTimeout(async () => {
    try {
      const response = await fetch(`/api/job/${state.jobId}`);
      const job = await response.json();
      if (!response.ok) throw new Error(job.error || "job vanished");

      updateProgress(job);

      if (job.state === "done") {
        state.report = job.report;
        renderReport(job.report);
        $("runDemo").disabled = false;
        setStatus("engine idle");
        setHint("Analysis complete. Streaming annotated output above.");
        return;
      }
      if (job.state === "error") {
        $("runDemo").disabled = false;
        setStatus("engine error");
        setHint(job.error || "job failed", true);
        return;
      }
      pollJob();
    } catch (error) {
      $("runDemo").disabled = false;
      setStatus("engine idle");
      setHint(error.message, true);
    }
  }, 900);
}

function updateProgress(job) {
  const pct = Math.round((job.progress || 0) * 100);
  $("progressBar").style.width = `${pct}%`;
  $("progressLabel").textContent = job.state === "queued" ? "queued" : `${pct}% · ${job.state}`;
  $("progressCount").textContent = `${job.processed || 0} / ${job.total || state.frames} frames`;
  $("frameStats").textContent =
    `${job.active || 0} tracked · ${job.identities || 0} identities · ${job.recoveries || 0} re-acquired`;
  if (job.state === "running") {
    setStatus(`tracking · ${job.identities || 0} ids`, true);
    setHint(`Processing frame ${job.processed} of ${job.total}…`);
  }
}

/* ------------------------------------------------------------------- render */
function renderReport(report) {
  const summary = report.summary;
  const runtime = report.runtime;

  $("mIdentities").textContent = summary.identities;
  $("mRecoveries").textContent = summary.recoveries;
  $("mOcclusion").textContent = summary.max_occlusion_frames ? `${summary.max_occlusion_frames}` : "0";
  $("mRejected").textContent = (summary.rejected_recoveries || 0) + (summary.duplicates_suppressed || 0);
  $("mFps").textContent = fmt(runtime.processing_fps, 1);
  $("mActive").textContent = summary.active;

  $("videoMeta").textContent =
    `${report.video.path} · ${report.video.width}×${report.video.height} · ${report.video.frames_processed} frames`;
  $("footStats").textContent =
    `${summary.identities} identities · ${summary.recoveries} verified re-acquisitions · ${runtime.seconds}s runtime`;

  const stream = $("stream");
  stream.src = `/media/${report.job_id || state.jobId}/stream.mjpg?t=${Date.now()}`;
  stream.classList.add("on");
  $("videoPlaceholder").style.display = "none";

  const download = $("downloadLink");
  download.href = `/media/${report.job_id || state.jobId}/download.mp4`;
  download.hidden = false;

  renderTimeline(report);
  renderEvents(report);
  renderIdentities(report);
}

function renderTimeline(report) {
  const host = $("timeline");
  const total = Math.max(1, report.video.frames_processed);
  const identities = (report.timeline || []).slice(0, 30);
  const recoveryFrames = (report.events || [])
    .filter((e) => e.type === "IDENTITY_RECOVERED")
    .reduce((acc, e) => {
      (acc[e.id] = acc[e.id] || []).push(e.frame);
      return acc;
    }, {});

  if (!identities.length) {
    host.innerHTML = '<div class="empty">No identities were confirmed in this sequence.</div>';
    return;
  }

  $("timelineMeta").textContent = `${identities.length} identities · ${total} frames`;

  host.innerHTML = identities
    .map((identity) => {
      const segments = (identity.segments || [])
        .map(([start, end]) => {
          const left = ((start - 1) / total) * 100;
          const width = Math.max(((end - start + 1) / total) * 100, 0.7);
          return `<div class="tl-seg" style="left:${left}%;width:${width}%"></div>`;
        })
        .join("");
      const dots = (recoveryFrames[identity.id] || [])
        .map((frame) => `<div class="tl-dot" style="left:${((frame - 1) / total) * 100}%"></div>`)
        .join("");
      return `<div class="tl-row">
        <span class="tl-id">${identity.id}</span>
        <div class="tl-track">${segments}${dots}</div>
      </div>`;
    })
    .join("");
}

function renderEvents(report) {
  const host = $("events");
  const events = (report.events || []).slice(-60).reverse();
  if (!events.length) {
    host.innerHTML = '<div class="empty">No identity events recorded.</div>';
    return;
  }
  host.innerHTML = events
    .map(
      (event) => `<div class="event t-${event.type}">
        <span class="f">f${event.frame}</span>
        <span class="type">${event.type.replace("IDENTITY_", "")}</span>
        <span class="id">${event.id}</span>
        <span class="detail" title="${(event.detail || "").replace(/"/g, "&quot;")}">
          ${event.confidence ? event.confidence.toFixed(2) + " · " : ""}${event.detail || ""}
        </span>
      </div>`
    )
    .join("");
}

function renderIdentities(report) {
  const body = document.querySelector("#identityTable tbody");
  const rows = (report.summary.identities_list || [])
    .slice()
    .sort((a, b) => b.hits - a.hits);
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="8" class="empty">No identities confirmed.</td></tr>';
    return;
  }
  body.innerHTML = rows
    .map(
      (row) => `<tr class="${row.recoveries ? "recovered" : ""}">
        <td>${row.id}</td>
        <td><span class="state-pill ${row.state}">${row.state}</span></td>
        <td>${row.first_frame}</td>
        <td>${row.last_frame}</td>
        <td>${row.hits}</td>
        <td>${row.gallery}</td>
        <td>${row.recoveries}</td>
        <td>${row.quality.toFixed(2)}</td>
      </tr>`
    )
    .join("");
}

/* ---------------------------------------------------------------- benchmark */
$("runBench").addEventListener("click", async () => {
  $("benchStatus").textContent = "Rendering scene and scoring CLEAR-MOT / IDF1 …";
  try {
    await fetch("/api/benchmark", { method: "POST" });
    pollBenchmark();
  } catch (error) {
    $("benchStatus").textContent = error.message;
  }
});

async function pollBenchmark() {
  try {
    const response = await fetch("/api/benchmark");
    const data = await response.json();
    if (data.state === "done") {
      state.benchmark = data.result;
      renderBenchmark(data.result);
      return;
    }
    if (data.state === "error") {
      $("benchStatus").textContent = data.error;
      return;
    }
    setTimeout(pollBenchmark, 1200);
  } catch (error) {
    $("benchStatus").textContent = error.message;
  }
}

function renderBenchmark(result) {
  const rows = [
    ["IDF1", "idf1", 3, true],
    ["MOTA", "mota", 3, true],
    ["Precision", "precision", 3, true],
    ["Recall", "recall", 3, true],
    ["ID switches", "id_switches", 0, false],
    ["Fragments", "fragments", 0, false],
    ["Unique identities", "unique_predictions", 0, false],
  ];

  const body = document.querySelector("#benchTable tbody");
  body.innerHTML = rows
    .map(([label, key, digits, higher]) => {
      const base = result.baseline[key];
      const ours = result.argus[key];
      const better = higher ? ours > base : ours < base;
      return `<tr>
        <td>${label}</td>
        <td>${Number(base).toFixed(digits)}</td>
        <td class="${better ? "better" : "worse"}">${Number(ours).toFixed(digits)}</td>
      </tr>`;
    })
    .join("");

  const seq = result.sequence;
  $("benchStatus").textContent =
    `${seq.frames} frames · ${seq.objects} actors · ${seq.resolution} · identical detections to both trackers`;
  $("benchNote").textContent =
    `${result.argus.recoveries} verified re-acquisitions, longest held identity gap ${result.argus.max_occlusion_frames} frames. ` +
    `Stressors: ${seq.stressors.join("; ")}.`;
}

/* --------------------------------------------------------------- live camera */
let camPoll = null;

async function camAction(path) {
  const response = await fetch(path, { method: "POST" });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `camera request failed (${response.status})`);
  return data;
}

function camStats(status) {
  $("camStats").textContent =
    `${status.frames || 0} frames · ${fmt(status.fps, 1)} fps · ${status.active || 0} tracked · ` +
    `${status.identities || 0} identities · ${status.recoveries || 0} re-acquired`;
}

$("camStart").addEventListener("click", async () => {
  $("camStart").disabled = true;
  $("camStatus").textContent = "opening the camera and loading the detector…";
  try {
    attachCamera(await camAction("/api/camera/start"));
  } catch (error) {
    $("camStatus").textContent = error.message;
    $("camStart").disabled = false;
  }
});

$("camStop").addEventListener("click", async () => {
  try {
    await camAction("/api/camera/stop");
  } catch (error) {
    /* the stream is detached either way */
  }
  detachCamera("camera idle");
});

function attachCamera(status) {
  const img = $("camStream");
  img.src = `/media/camera/stream.mjpg?t=${Date.now()}`;
  img.classList.add("on");
  $("camPlaceholder").style.display = "none";
  $("camStop").disabled = false;
  const size = status.resolution || [];
  $("camStatus").textContent =
    status.state === "running"
      ? `streaming ${size[0] || "?"}×${size[1] || "?"} · device ${status.index ?? 0}`
      : status.error || status.state;
  camStats(status);
  clearTimeout(camPoll);
  camPoll = setTimeout(pollCamera, 1500);
}

function detachCamera(reason) {
  clearTimeout(camPoll);
  const img = $("camStream");
  img.classList.remove("on");
  img.removeAttribute("src");
  $("camPlaceholder").style.display = "grid";
  $("camStart").disabled = false;
  $("camStop").disabled = true;
  $("camStatus").textContent = reason;
  $("camStats").textContent = "—";
}

async function pollCamera() {
  try {
    const response = await fetch("/api/camera/status");
    const status = await response.json();
    if (status.state === "running") {
      camStats(status);
      camPoll = setTimeout(pollCamera, 1500);
      return;
    }
    detachCamera(status.state === "error" ? status.error || "camera error" : "camera idle");
  } catch (error) {
    detachCamera(error.message);
  }
}

/* ---------------------------------------------------------------- bootstrap */
(async function boot() {
  try {
    const camera = await (await fetch("/api/camera/status")).json();
    if (camera.state === "running") attachCamera(camera);
  } catch (error) {
    /* camera is optional */
  }

  try {
    const response = await fetch("/api/benchmark");
    const data = await response.json();
    if (data.state === "done" && data.result) {
      renderBenchmark(data.result);
    } else if (data.state === "running") {
      pollBenchmark();
    }
  } catch (error) {
    /* dashboard still works without the benchmark */
  }

  try {
    const response = await fetch("/api/latest");
    const data = await response.json();
    if (data.job_id) {
      state.jobId = data.job_id;
      const job = await (await fetch(`/api/job/${data.job_id}`)).json();
      if (job.report) {
        state.report = job.report;
        renderReport(job.report);
        setHint("Showing the most recent sequence. Run again to overwrite.");
      }
    }
  } catch (error) {
    /* no previous run */
  }
})();
