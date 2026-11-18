/**
 * PathSnap — Map-First Workstation Application Controller
 * Pure Local Offline Leaflet.js Vector Mapping Engine (100% Offline / Zero Remote CDN)
 * Supports Student Test, Evaluator Holdout Test, and Training Partitions.
 */

(function () {
  'use strict';

  // Application State
  const state = {
    apiBase: window.location.origin,
    activeDataset: 'student_test',
    activeTrip: null,
    currentTripData: null,
    currentMatchResult: null,
    map: null,
    layers: {
      network: null,
      gps: null,
      prediction: null,
    },
    bounds: null,
  };

  // DOM Elements Cache
  const els = {};

  function initElements() {
    els.datasetSelect = document.getElementById('dataset-select');
    els.tripSelect = document.getElementById('trip-select');
    els.btnRun = document.getElementById('btn-run-matching');
    els.btnFit = document.getElementById('btn-fit-map');
    els.btnDrawer = document.getElementById('btn-toggle-drawer');
    els.drawer = document.getElementById('details-drawer');
    els.drawerClose = document.getElementById('btn-close-drawer');

    // Layer checkboxes
    els.chkNetwork = document.getElementById('chk-layer-network');
    els.chkGps = document.getElementById('chk-layer-gps');
    els.chkPred = document.getElementById('chk-layer-pred');

    // Floating map controls
    els.btnZoomIn = document.getElementById('map-btn-zoom-in');
    els.btnZoomOut = document.getElementById('map-btn-zoom-out');
    els.btnFloatingFit = document.getElementById('map-btn-fit');
    els.btnFloatingReset = document.getElementById('map-btn-reset');

    // Telemetry displays
    els.infoTripId = document.getElementById('info-trip-id');
    els.infoDataset = document.getElementById('info-dataset');
    els.infoGpsCount = document.getElementById('info-gps-count');
    els.infoArcsCount = document.getElementById('info-arcs-count');

    // Status pill
    els.statusPill = document.getElementById('status-pill');
    els.statusPillText = document.getElementById('status-pill-text');

    // Metric tiles
    els.valTop1 = document.getElementById('metric-val-top1');
    els.valCont = document.getElementById('metric-val-cont');
    els.valF1 = document.getElementById('metric-val-f1');
    els.valLatency = document.getElementById('metric-val-latency');

    // Bottom bar elements
    els.bottomStatus = document.getElementById('bottom-status-text');
    els.bottomTrip = document.getElementById('bottom-selected-trip');
    els.bottomDataset = document.getElementById('bottom-dataset');
    els.bottomLatency = document.getElementById('bottom-latency');

    // Drawer components
    els.evalSummaryBox = document.getElementById('eval-summary-box');
    els.drawerJsonBox = document.getElementById('drawer-raw-json');
  }

  // ==========================================================================
  // 1. OFFLINE LEAFLET MAP INITIALIZATION (Zero Remote Tiles / Pure Vector)
  // ==========================================================================
  function initMap() {
    // Initialize map on clean offline background
    state.map = L.map('map', {
      zoomControl: false,
      attributionControl: false,
      preferCanvas: true,
      minZoom: 3,
      maxZoom: 19,
    }).setView([56.81, 35.87], 13);

    // Metric scale control at bottom-left
    L.control.scale({ imperial: false, position: 'bottomleft' }).addTo(state.map);

    // Initialize dedicated layer groups directly onto the map
    state.layers.network = L.layerGroup().addTo(state.map);
    state.layers.gps = L.layerGroup().addTo(state.map);
    state.layers.prediction = L.layerGroup().addTo(state.map);

    // Window resize handler to prevent zero-height or clipping bugs
    window.addEventListener('resize', () => {
      if (state.map) state.map.invalidateSize();
    });
  }

  // ==========================================================================
  // 2. DATASET & TRIP MANAGEMENT
  // ==========================================================================
  async function loadDatasets() {
    try {
      const res = await fetch(`${state.apiBase}/api/datasets`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (els.datasetSelect && data.datasets) {
        els.datasetSelect.innerHTML = '';
        data.datasets.forEach((ds) => {
          const opt = document.createElement('option');
          opt.value = ds.id;
          opt.textContent = `${ds.name} (${ds.count} trips)`;
          els.datasetSelect.appendChild(opt);
        });
        els.datasetSelect.value = state.activeDataset;
      }
      await loadTripsForDataset(state.activeDataset);
    } catch (err) {
      console.warn('Dataset enumeration fallback:', err);
      await loadTripsForDataset(state.activeDataset);
    }
  }

  async function loadTripsForDataset(datasetId) {
    state.activeDataset = datasetId;
    if (els.infoDataset) els.infoDataset.textContent = datasetId;
    if (els.bottomDataset) els.bottomDataset.textContent = datasetId;
    setStatus('Loading trips...', 'running');

    try {
      const res = await fetch(`${state.apiBase}/api/trips?dataset=${datasetId}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const trips = data.trips || [];

      if (els.tripSelect) {
        els.tripSelect.innerHTML = '';
        trips.forEach((t) => {
          const opt = document.createElement('option');
          opt.value = t;
          opt.textContent = `Trip ${t}`;
          els.tripSelect.appendChild(opt);
        });

        if (trips.length > 0) {
          if (datasetId === 'student_test' && trips.includes('00000052')) {
            els.tripSelect.value = '00000052';
          } else {
            els.tripSelect.value = trips[0];
          }
          onTripChange();
        }
      }
    } catch (err) {
      setStatus(`Error: ${err.message}`, 'error');
    }
  }

  function onTripChange() {
    if (!els.tripSelect) return;
    const selected = els.tripSelect.value;
    state.activeTrip = selected;

    if (els.infoTripId) els.infoTripId.textContent = selected;
    if (els.bottomTrip) els.bottomTrip.textContent = selected;

    // Reset metric tiles until Run is executed
    if (els.valTop1) els.valTop1.textContent = '—';
    if (els.valCont) els.valCont.textContent = '—';
    if (els.valF1) els.valF1.textContent = '—';
    if (els.valLatency) els.valLatency.textContent = '—';

    // Load actual road network and GPS points immediately for inspection
    loadTripData(selected);
  }

  // ==========================================================================
  // 3. LOAD TRIP DATA (ROAD NETWORK & GPS OBSERVATIONS)
  // ==========================================================================
  async function loadTripData(tripId) {
    if (!tripId) return;
    setStatus(`Loading road network & GPS for Trip ${tripId}...`, 'running');

    try {
      const res = await fetch(`${state.apiBase}/api/trip/${tripId}?dataset=${state.activeDataset}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      state.currentTripData = data;
      state.currentMatchResult = null;

      // Update metadata displays
      if (els.infoTripId) els.infoTripId.textContent = data.trip_id;
      if (els.infoDataset) els.infoDataset.textContent = data.dataset;
      if (els.infoGpsCount) els.infoGpsCount.textContent = (data.n_gps || 0).toLocaleString();
      if (els.infoArcsCount) els.infoArcsCount.textContent = (data.n_road_arcs || 0).toLocaleString();

      // Clear all existing map layers
      state.layers.network.clearLayers();
      state.layers.gps.clearLayers();
      state.layers.prediction.clearLayers();

      // 1. Render Genuine Road Network (crisp neutral gray lines)
      if (data.network_segments && data.network_segments.length > 0) {
        L.polyline(data.network_segments, {
          color: '#94a3b8',
          weight: 1.5,
          opacity: 0.70,
        }).addTo(state.layers.network);
      }

      // 2. Render GPS Observations (blue markers + dashed trajectory)
      if (data.gps_points && data.gps_points.length > 0) {
        L.polyline(data.gps_points, {
          color: '#3b82f6',
          weight: 1.5,
          opacity: 0.45,
          dashArray: '3, 4',
        }).addTo(state.layers.gps);

        data.gps_points.forEach((pt, idx) => {
          const m = L.circleMarker(pt, {
            radius: 3.0,
            color: '#1d4ed8',
            fillColor: '#3b82f6',
            fillOpacity: 0.85,
            weight: 1.0,
          });
          m.bindTooltip(`GPS #${idx + 1}<br>Lat: ${pt[0].toFixed(5)}, Lon: ${pt[1].toFixed(5)}`);
          m.addTo(state.layers.gps);
        });
      }

      // Fit map bounds to trip extent
      state.bounds = data.bounds;
      if (data.bounds && data.bounds.length === 2) {
        state.map.fitBounds(data.bounds, { padding: [40, 40], maxZoom: 16 });
      }

      toggleLayerVisibility();
      setStatus(`Road network & GPS loaded (${data.network_segments.length} edges, ${data.n_gps} fixes)`, 'ready');
    } catch (err) {
      setStatus(`Load error: ${err.message}`, 'error');
    }
  }

  // ==========================================================================
  // 4. RUN MAP MATCHING (MLP + HMM VITERBI PREDICTION)
  // ==========================================================================
  async function runMapMatching() {
    const tripId = state.activeTrip || (els.tripSelect ? els.tripSelect.value : null);
    if (!tripId) return;

    setStatus(`Computing map match for Trip ${tripId}...`, 'running');
    if (els.btnRun) els.btnRun.disabled = true;

    try {
      const res = await fetch(`${state.apiBase}/api/match/${tripId}?dataset=${state.activeDataset}`, {
        method: 'POST',
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      state.currentMatchResult = data;

      // If road network wasn't already loaded, render it now
      if (state.layers.network.getLayers().length === 0 && data.network_segments) {
        L.polyline(data.network_segments, {
          color: '#94a3b8',
          weight: 1.5,
          opacity: 0.70,
        }).addTo(state.layers.network);
      }
      if (state.layers.gps.getLayers().length === 0 && data.gps_points) {
        L.polyline(data.gps_points, {
          color: '#3b82f6',
          weight: 1.5,
          opacity: 0.45,
          dashArray: '3, 4',
        }).addTo(state.layers.gps);
        data.gps_points.forEach((pt, idx) => {
          const m = L.circleMarker(pt, {
            radius: 3.0,
            color: '#1d4ed8',
            fillColor: '#3b82f6',
            fillOpacity: 0.85,
            weight: 1.0,
          });
          m.bindTooltip(`GPS #${idx + 1}`);
          m.addTo(state.layers.gps);
        });
      }

      // Render Predicted Snapped Route (bold red polyline + start/end markers)
      state.layers.prediction.clearLayers();
      if (data.predicted_segments && data.predicted_segments.length > 0) {
        L.polyline(data.predicted_segments, {
          color: '#dc2626',
          weight: 4.5,
          opacity: 0.95,
          lineJoin: 'round',
          lineCap: 'round',
        }).addTo(state.layers.prediction);

        // Start marker (green)
        if (data.start_point) {
          L.circleMarker(data.start_point, {
            radius: 6.5,
            color: '#15803d',
            fillColor: '#22c55e',
            fillOpacity: 1.0,
            weight: 2.5,
          }).bindTooltip('Trip Start (Snapped)', { permanent: false }).addTo(state.layers.prediction);
        }

        // End marker (red)
        if (data.end_point) {
          L.circleMarker(data.end_point, {
            radius: 6.5,
            color: '#991b1b',
            fillColor: '#ef4444',
            fillOpacity: 1.0,
            weight: 2.5,
          }).bindTooltip('Trip End (Snapped)', { permanent: false }).addTo(state.layers.prediction);
        }
      }

      updateTelemetryAndMetrics(data);
      toggleLayerVisibility();
      setStatus(`Match complete in ${data.inference_time_s}s`, 'ready');
    } catch (err) {
      setStatus(`Inference Error: ${err.message}`, 'error');
    } finally {
      if (els.btnRun) els.btnRun.disabled = false;
    }
  }

  function toggleLayerVisibility() {
    if (!state.map) return;
    if (els.chkNetwork) {
      if (els.chkNetwork.checked) state.map.addLayer(state.layers.network);
      else state.map.removeLayer(state.layers.network);
    }
    if (els.chkGps) {
      if (els.chkGps.checked) state.map.addLayer(state.layers.gps);
      else state.map.removeLayer(state.layers.gps);
    }
    if (els.chkPred) {
      if (els.chkPred.checked) state.map.addLayer(state.layers.prediction);
      else state.map.removeLayer(state.layers.prediction);
    }
  }

  function updateTelemetryAndMetrics(data) {
    if (els.infoGpsCount) els.infoGpsCount.textContent = (data.n_gps || 0).toLocaleString();
    if (els.infoArcsCount) els.infoArcsCount.textContent = (data.n_road_arcs || 0).toLocaleString();
    if (els.valLatency) els.valLatency.textContent = `${data.inference_time_s}s`;
    if (els.bottomLatency) els.bottomLatency.textContent = `Latency: ${data.inference_time_s}s`;

    const m = data.metrics || {};
    const d = data.diagnostics || {};

    // Topological continuity
    const contVal = m.hmm_continuity !== undefined ? m.hmm_continuity : d.predicted_continuity;
    if (els.valCont && contVal !== undefined) {
      els.valCont.textContent = `${(contVal * 100).toFixed(1)}%`;
    }

    // Top-1 Accuracy
    if (els.valTop1) {
      if (m.hmm_top1_acc !== undefined) {
        els.valTop1.textContent = `${(m.hmm_top1_acc * 100).toFixed(1)}%`;
      } else {
        els.valTop1.textContent = '—';
      }
    }

    // Sequence Arc F1
    if (els.valF1) {
      if (m.hmm_arc_f1 !== undefined) {
        els.valF1.textContent = `${(m.hmm_arc_f1 * 100).toFixed(1)}%`;
      } else {
        els.valF1.textContent = '—';
      }
    }

    // Update raw JSON in drawer
    if (els.drawerJsonBox) {
      els.drawerJsonBox.textContent = JSON.stringify(
        {
          trip_id: data.trip_id,
          dataset: data.dataset,
          observations: data.n_gps,
          road_arcs: data.n_road_arcs,
          inference_time_s: data.inference_time_s,
          diagnostics: data.diagnostics,
          metrics: data.metrics,
          predicted_unique_arcs_count: (data.predicted_arc_sequence || []).length,
        },
        null,
        2
      );
    }
  }

  function setStatus(msg, type = 'ready') {
    if (els.statusPillText) els.statusPillText.textContent = msg;
    if (els.bottomStatus) els.bottomStatus.textContent = `Status: ${msg}`;
    if (els.statusPill) {
      els.statusPill.className = `status-pill ${type}`;
    }
  }

  // ==========================================================================
  // 5. EVALUATOR SUMMARY & DRAWER CONTROLS
  // ==========================================================================
  async function loadEvaluatorSummary() {
    try {
      const res = await fetch(`${state.apiBase}/api/evaluator/summary`);
      if (!res.ok) return;
      const data = await res.json();
      if (els.evalSummaryBox && data.metrics_summary) {
        const m = data.metrics_summary;
        els.evalSummaryBox.innerHTML = `
          <table class="table-mini">
            <tr><td>Evaluated Trips</td><td>${data.evaluated_trips_count} holdout trips</td></tr>
            <tr><td>HMM Top-1 Acc (Mean)</td><td>${(m.hmm_top1_accuracy.mean * 100).toFixed(2)}%</td></tr>
            <tr><td>Classifier Top-1 Acc</td><td>${(m.clf_top1_accuracy.mean * 100).toFixed(2)}%</td></tr>
            <tr><td>Route Continuity</td><td>${(m.hmm_route_continuity.mean * 100).toFixed(2)}%</td></tr>
            <tr><td>Sequence Arc F1</td><td>${(m.hmm_arc_f1.mean * 100).toFixed(2)}%</td></tr>
            <tr><td>Sequence Jaccard</td><td>${(m.hmm_arc_jaccard.mean * 100).toFixed(2)}%</td></tr>
            <tr><td>LCS Sequence Similarity</td><td>${(m.hmm_lcs_f1.mean * 100).toFixed(2)}%</td></tr>
            <tr><td>Mean Latency</td><td>${m.inference_time_s.mean.toFixed(3)} s/trip</td></tr>
          </table>
        `;
      }
    } catch (e) {
      console.warn('Evaluator summary load:', e);
    }
  }

  function toggleDrawer(open) {
    if (!els.drawer) return;
    const shouldOpen = open !== undefined ? open : !els.drawer.classList.contains('open');
    if (shouldOpen) {
      els.drawer.classList.add('open');
      loadEvaluatorSummary();
    } else {
      els.drawer.classList.remove('open');
    }
  }

  function fitTripBounds() {
    if (state.map && state.bounds && state.bounds.length === 2) {
      state.map.fitBounds(state.bounds, { padding: [40, 40], maxZoom: 16 });
    }
  }

  // ==========================================================================
  // 6. EVENT BINDINGS & BOOTSTRAP
  // ==========================================================================
  function bindEvents() {
    if (els.datasetSelect) {
      els.datasetSelect.addEventListener('change', () => {
        loadTripsForDataset(els.datasetSelect.value);
      });
    }

    if (els.tripSelect) {
      els.tripSelect.addEventListener('change', onTripChange);
    }

    if (els.btnRun) {
      els.btnRun.addEventListener('click', runMapMatching);
    }

    if (els.btnFit) {
      els.btnFit.addEventListener('click', fitTripBounds);
    }

    if (els.btnDrawer) {
      els.btnDrawer.addEventListener('click', () => toggleDrawer());
    }

    if (els.drawerClose) {
      els.drawerClose.addEventListener('click', () => toggleDrawer(false));
    }

    // Floating map controls
    if (els.btnZoomIn) {
      els.btnZoomIn.addEventListener('click', () => {
        if (state.map) state.map.zoomIn();
      });
    }
    if (els.btnZoomOut) {
      els.btnZoomOut.addEventListener('click', () => {
        if (state.map) state.map.zoomOut();
      });
    }
    if (els.btnFloatingFit) {
      els.btnFloatingFit.addEventListener('click', fitTripBounds);
    }
    if (els.btnFloatingReset) {
      els.btnFloatingReset.addEventListener('click', fitTripBounds);
    }

    // Layer toggles
    [els.chkNetwork, els.chkGps, els.chkPred].forEach((chk) => {
      if (chk) chk.addEventListener('change', toggleLayerVisibility);
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        runMapMatching();
      } else if (e.key === 'f' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        fitTripBounds();
      }
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    initElements();
    initMap();
    bindEvents();
    loadDatasets();
    loadEvaluatorSummary();
  });
})();
