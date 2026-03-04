import { useState, useEffect, useCallback } from "react";
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine
} from "recharts";

// ─── Config ────────────────────────────────────────────────────────────────
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

// ─── Utility ────────────────────────────────────────────────────────────────
const fmt = (v) => typeof v === "number" ? v.toFixed(3) : v;
const fmtTime = (iso) => {
  const d = new Date(iso);
  return `${d.getMonth()+1}/${d.getDate()} ${String(d.getHours()).padStart(2,"0")}:00`;
};

// ─── Custom Tooltip ──────────────────────────────────────────────────────────
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#0d1f2d",
      border: "1px solid rgba(0,229,255,0.3)",
      borderRadius: 10,
      padding: "10px 14px",
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: 12,
    }}>
      <p style={{ color: "#5a9ab5", marginBottom: 6, fontSize: 11 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color || "#00e5ff", margin: "2px 0" }}>
          {p.name}: <strong>{typeof p.value === "number" ? p.value.toFixed(3) : p.value} kW</strong>
        </p>
      ))}
    </div>
  );
};

// ─── Metric Card ─────────────────────────────────────────────────────────────
const MetricCard = ({ label, value, unit, color = "#00e5ff", delay = 0 }) => (
  <div style={{
    background: "#0a1828",
    border: `1px solid ${color}22`,
    borderLeft: `3px solid ${color}`,
    borderRadius: 10,
    padding: "16px 20px",
    animation: `fadeUp 0.5s ${delay}s ease both`,
  }}>
    <div style={{ fontSize: 10, letterSpacing: 2, color: "#4a7a90", textTransform: "uppercase", marginBottom: 8 }}>{label}</div>
    <div style={{ fontSize: 28, fontFamily: "'Syne', sans-serif", fontWeight: 800, color }}>
      {value}<span style={{ fontSize: 14, color: "#4a7a90", marginLeft: 4 }}>{unit}</span>
    </div>
  </div>
);

// ─── Attention Heatmap ────────────────────────────────────────────────────────
const AttentionHeatmap = ({ weights }) => {
  if (!weights?.length) return null;
  return (
    <div>
      <div style={{ fontSize: 11, color: "#4a7a90", letterSpacing: 2, marginBottom: 10, textTransform: "uppercase" }}>
        Attention Weights — What the model focuses on
      </div>
      <div style={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
        {weights.map((w, i) => (
          <div
            key={i}
            title={`Patch ${i + 1}: ${(w * 100).toFixed(1)}%`}
            style={{
              width: 20, height: 20,
              borderRadius: 4,
              background: `rgba(0,229,255,${Math.max(0.05, w)})`,
              border: "1px solid rgba(0,229,255,0.1)",
              cursor: "default",
            }}
          />
        ))}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
        <span style={{ fontSize: 10, color: "#2a4a5a" }}>Low attention</span>
        <span style={{ fontSize: 10, color: "#00e5ff" }}>High attention</span>
      </div>
    </div>
  );
};

// ─── Model Compare Table ──────────────────────────────────────────────────────
const CompareTable = ({ data }) => {
  const models = Object.keys(data || {});
  const horizons = [24, 48, 96, 168];
  const [selH, setSelH] = useState(24);

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        {horizons.map(h => (
          <button key={h} onClick={() => setSelH(h)} style={{
            padding: "6px 14px",
            borderRadius: 20,
            border: `1px solid ${selH === h ? "#00e5ff" : "rgba(0,229,255,0.15)"}`,
            background: selH === h ? "rgba(0,229,255,0.15)" : "transparent",
            color: selH === h ? "#00e5ff" : "#4a7a90",
            fontSize: 12,
            cursor: "pointer",
            fontFamily: "'JetBrains Mono', monospace",
          }}>{h}h</button>
        ))}
      </div>
      <table style={{ width: "100%", borderCollapse: "separate", borderSpacing: "0 6px" }}>
        <thead>
          <tr>
            {["Model", "MAE", "RMSE", "MAPE"].map(h => (
              <th key={h} style={{ fontSize: 10, letterSpacing: 2, color: "#4a7a90", textAlign: "left", padding: "0 12px 8px", textTransform: "uppercase" }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map(model => {
            const row = data[model]?.find(r => r.horizon === selH) || {};
            const isBest = model === "PatchTST";
            return (
              <tr key={model}>
                {[
                  <td key="name" style={{ padding: "10px 12px", background: "#0a1828", borderRadius: "8px 0 0 8px", borderLeft: `2px solid ${isBest ? "#00e5ff" : "transparent"}`, color: isBest ? "#00e5ff" : "#c8dde8", fontWeight: isBest ? 600 : 400, fontSize: 13 }}>
                    {model} {isBest && <span style={{ fontSize: 10, color: "#00ff9d", marginLeft: 6 }}>★ BEST</span>}
                  </td>,
                  ...["mae", "rmse", "mape"].map(k => (
                    <td key={k} style={{ padding: "10px 12px", background: "#0a1828", borderRadius: k === "mape" ? "0 8px 8px 0" : 0, fontSize: 13, color: isBest ? "#00ff9d" : "#8aabbb" }}>
                      {row[k]}{k === "mape" ? "%" : ""}
                    </td>
                  ))
                ]}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [horizon, setHorizon] = useState(24);
  const [forecastData, setForecastData] = useState(null);
  const [compareData, setCompareData] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("forecast");

  const runForecast = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/forecast/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ horizon }),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      setForecastData(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [horizon]);

  useEffect(() => {
    runForecast();
    fetch(`${API_BASE}/compare/`).then(r => r.json()).then(d => setCompareData(d.models));
    fetch(`${API_BASE}/model-info/`).then(r => r.json()).then(setModelInfo);
  }, []);

  // Build chart data
  const chartData = forecastData ? [
    ...forecastData.historical.map((v, i) => ({
      time: fmtTime(forecastData.historical_timestamps[i]),
      actual: +v.toFixed(3),
      upper_hist: +forecastData.historical_upper[i].toFixed(3),
      lower_hist: +forecastData.historical_lower[i].toFixed(3),
      type: "historical",
    })),
    ...forecastData.predictions.map((v, i) => ({
      time: fmtTime(forecastData.forecast_timestamps[i]),
      forecast: +v.toFixed(3),
      upper: +forecastData.upper_bound[i].toFixed(3),
      lower: +forecastData.lower_bound[i].toFixed(3),
      type: "forecast",
    }))
  ] : [];

  const splitIdx = forecastData?.historical?.length ?? 0;

  return (
    <div style={{
      minHeight: "100vh",
      background: "#050d14",
      color: "#c8dde8",
      fontFamily: "'JetBrains Mono', monospace",
    }}>
      {/* Grid bg */}
      <div style={{
        position: "fixed", inset: 0,
        backgroundImage: "linear-gradient(rgba(0,229,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(0,229,255,0.025) 1px, transparent 1px)",
        backgroundSize: "40px 40px",
        pointerEvents: "none",
      }} />

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "40px 24px", position: "relative" }}>

        {/* Header */}
        <div style={{ marginBottom: 48, animation: "fadeDown 0.7s ease both" }}>
          <div style={{ fontSize: 10, letterSpacing: 4, color: "#00e5ff", marginBottom: 12, display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ width: 28, height: 1, background: "#00e5ff", display: "inline-block" }} />
            DEEP LEARNING · TIME SERIES · TRANSFORMER
          </div>
          <h1 style={{
            fontFamily: "'Syne', sans-serif",
            fontSize: "clamp(28px, 5vw, 52px)",
            fontWeight: 800,
            lineHeight: 1.1,
            background: "linear-gradient(135deg, #fff 0%, #00e5ff 50%, #00ff9d 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            marginBottom: 12,
          }}>
            Energy Consumption<br />Forecasting
          </h1>
          <p style={{ color: "#4a7a90", fontSize: 13, maxWidth: 520, lineHeight: 1.7 }}>
            PatchTST Transformer trained on UCI Household Power Consumption. Multi-horizon predictions with confidence intervals and attention visualization.
          </p>
        </div>

        {/* Controls */}
        <div style={{
          display: "flex", alignItems: "center", gap: 12, marginBottom: 32,
          flexWrap: "wrap",
          animation: "fadeUp 0.6s 0.1s ease both",
        }}>
          <span style={{ fontSize: 11, color: "#4a7a90", letterSpacing: 2 }}>HORIZON:</span>
          {[24, 48, 96, 168].map(h => (
            <button key={h} onClick={() => setHorizon(h)} style={{
              padding: "8px 18px",
              borderRadius: 8,
              border: `1px solid ${horizon === h ? "#00e5ff" : "rgba(0,229,255,0.15)"}`,
              background: horizon === h ? "rgba(0,229,255,0.12)" : "transparent",
              color: horizon === h ? "#00e5ff" : "#4a7a90",
              fontSize: 13,
              cursor: "pointer",
              fontFamily: "'JetBrains Mono', monospace",
              transition: "all 0.2s",
            }}>
              {h}h {["", "", "4-day", "weekly"][([24,48,96,168].indexOf(h))]}
            </button>
          ))}
          <button onClick={runForecast} disabled={loading} style={{
            marginLeft: "auto",
            padding: "9px 22px",
            borderRadius: 8,
            border: "1px solid #00e5ff",
            background: loading ? "rgba(0,229,255,0.05)" : "rgba(0,229,255,0.15)",
            color: "#00e5ff",
            fontSize: 13,
            cursor: loading ? "not-allowed" : "pointer",
            fontFamily: "'JetBrains Mono', monospace",
            transition: "all 0.2s",
          }}>
            {loading ? "⟳ Running..." : "▶ Run Forecast"}
          </button>
        </div>

        {error && (
          <div style={{ background: "rgba(255,80,80,0.1)", border: "1px solid rgba(255,80,80,0.3)", borderRadius: 10, padding: "12px 16px", marginBottom: 24, color: "#ff8080", fontSize: 13 }}>
            ⚠ {error} — Make sure the Django backend is running on port 8000.
          </div>
        )}

        {/* Metrics */}
        {forecastData && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12, marginBottom: 32 }}>
            <MetricCard label="MAE" value={fmt(forecastData.metrics?.mae)} unit="kW" color="#00e5ff" delay={0} />
            <MetricCard label="RMSE" value={fmt(forecastData.metrics?.rmse)} unit="kW" color="#00ff9d" delay={0.05} />
            <MetricCard label="MAPE" value={fmt(forecastData.metrics?.mape)} unit="%" color="#ffd166" delay={0.1} />
            <MetricCard label="Horizon" value={horizon} unit="hrs" color="#ff6b35" delay={0.15} />
          </div>
        )}

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, marginBottom: 24, borderBottom: "1px solid rgba(0,229,255,0.1)", paddingBottom: 0 }}>
          {["forecast", "compare", "architecture"].map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)} style={{
              padding: "10px 20px",
              background: "transparent",
              border: "none",
              borderBottom: `2px solid ${activeTab === tab ? "#00e5ff" : "transparent"}`,
              color: activeTab === tab ? "#00e5ff" : "#4a7a90",
              fontSize: 12,
              letterSpacing: 2,
              textTransform: "uppercase",
              cursor: "pointer",
              fontFamily: "'JetBrains Mono', monospace",
              transition: "all 0.2s",
            }}>{tab}</button>
          ))}
        </div>

        {/* Forecast Tab */}
        {activeTab === "forecast" && (
          <div style={{ animation: "fadeUp 0.4s ease both" }}>
            {/* Main chart */}
            <div style={{
              background: "#0a1828",
              border: "1px solid rgba(0,229,255,0.1)",
              borderRadius: 14,
              padding: "24px",
              marginBottom: 20,
            }}>
              <div style={{ fontSize: 12, letterSpacing: 2, color: "#4a7a90", textTransform: "uppercase", marginBottom: 20 }}>
                Forecast · Historical + Predicted with 95% Confidence Intervals
              </div>
              {loading ? (
                <div style={{ height: 320, display: "flex", alignItems: "center", justifyContent: "center", color: "#00e5ff" }}>
                  Running PatchTST inference...
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={340}>
                  <AreaChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                    <defs>
                      <linearGradient id="histGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#00e5ff" stopOpacity={0.15} />
                        <stop offset="95%" stopColor="#00e5ff" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="predGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#00ff9d" stopOpacity={0.2} />
                        <stop offset="95%" stopColor="#00ff9d" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,229,255,0.06)" />
                    <XAxis dataKey="time" tick={{ fill: "#3a6a80", fontSize: 10 }} tickLine={false} interval={Math.floor(chartData.length / 8)} />
                    <YAxis tick={{ fill: "#3a6a80", fontSize: 10 }} tickLine={false} axisLine={false} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 11, color: "#4a7a90" }} />
                    {splitIdx > 0 && <ReferenceLine x={chartData[splitIdx - 1]?.time} stroke="#ffd166" strokeDasharray="4 4" label={{ value: "NOW", fill: "#ffd166", fontSize: 10 }} />}
                    {/* Confidence band historical */}
                    <Area type="monotone" dataKey="upper_hist" stroke="none" fill="url(#histGrad)" name="CI (hist)" legendType="none" />
                    <Area type="monotone" dataKey="lower_hist" stroke="none" fill="#0a1828" name="CI (hist) lower" legendType="none" />
                    {/* Confidence band forecast */}
                    <Area type="monotone" dataKey="upper" stroke="none" fill="url(#predGrad)" name="CI (forecast)" legendType="none" />
                    <Area type="monotone" dataKey="lower" stroke="none" fill="#0a1828" name="CI lower" legendType="none" />
                    <Line type="monotone" dataKey="actual" stroke="#00e5ff" strokeWidth={2} dot={false} name="Actual" />
                    <Line type="monotone" dataKey="forecast" stroke="#00ff9d" strokeWidth={2} dot={false} strokeDasharray="6 3" name="Forecast" />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>

            {/* Attention heatmap */}
            {forecastData?.attention_weights && (
              <div style={{
                background: "#0a1828",
                border: "1px solid rgba(0,229,255,0.1)",
                borderRadius: 14,
                padding: "24px",
              }}>
                <AttentionHeatmap weights={forecastData.attention_weights} />
              </div>
            )}
          </div>
        )}

        {/* Compare Tab */}
        {activeTab === "compare" && (
          <div style={{ animation: "fadeUp 0.4s ease both" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              <div style={{ background: "#0a1828", border: "1px solid rgba(0,229,255,0.1)", borderRadius: 14, padding: 24 }}>
                <div style={{ fontSize: 12, letterSpacing: 2, color: "#4a7a90", textTransform: "uppercase", marginBottom: 20 }}>Benchmark Results</div>
                {compareData && <CompareTable data={compareData} />}
              </div>
              <div style={{ background: "#0a1828", border: "1px solid rgba(0,229,255,0.1)", borderRadius: 14, padding: 24 }}>
                <div style={{ fontSize: 12, letterSpacing: 2, color: "#4a7a90", textTransform: "uppercase", marginBottom: 20 }}>MAE by Horizon (24h)</div>
                {compareData && (
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={Object.entries(compareData).map(([name, rows]) => ({
                      name,
                      mae: rows.find(r => r.horizon === 24)?.mae,
                    }))} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,229,255,0.06)" />
                      <XAxis dataKey="name" tick={{ fill: "#3a6a80", fontSize: 11 }} tickLine={false} />
                      <YAxis tick={{ fill: "#3a6a80", fontSize: 11 }} tickLine={false} axisLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="mae" name="MAE" radius={[4, 4, 0, 0]}
                        fill="#00e5ff"
                        label={{ position: "top", fill: "#4a7a90", fontSize: 10 }}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Architecture Tab */}
        {activeTab === "architecture" && modelInfo && (
          <div style={{ animation: "fadeUp 0.4s ease both" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              <div style={{ background: "#0a1828", border: "1px solid rgba(0,229,255,0.1)", borderRadius: 14, padding: 24 }}>
                <div style={{ fontSize: 12, letterSpacing: 2, color: "#4a7a90", textTransform: "uppercase", marginBottom: 20 }}>Model Architecture</div>
                <div style={{ fontSize: 17, fontFamily: "'Syne', sans-serif", fontWeight: 700, color: "#00e5ff", marginBottom: 6 }}>{modelInfo.architecture}</div>
                <div style={{ fontSize: 12, color: "#4a7a90", marginBottom: 20, lineHeight: 1.6 }}>"{modelInfo.paper}"</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  {Object.entries(modelInfo.hyperparameters || {}).map(([k, v]) => (
                    <div key={k} style={{ background: "#050d14", borderRadius: 8, padding: "10px 14px", border: "1px solid rgba(0,229,255,0.08)" }}>
                      <div style={{ fontSize: 9, color: "#4a7a90", letterSpacing: 1, marginBottom: 4, textTransform: "uppercase" }}>{k.replace(/_/g, " ")}</div>
                      <div style={{ fontSize: 16, color: "#00ff9d", fontWeight: 600 }}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div style={{ background: "#0a1828", border: "1px solid rgba(0,229,255,0.1)", borderRadius: 14, padding: 24 }}>
                <div style={{ fontSize: 12, letterSpacing: 2, color: "#4a7a90", textTransform: "uppercase", marginBottom: 20 }}>Input Features</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {modelInfo.features?.map((f, i) => (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 14px", background: "#050d14", borderRadius: 8, border: "1px solid rgba(0,229,255,0.08)" }}>
                      <div style={{ width: 6, height: 6, borderRadius: "50%", background: `hsl(${i * 45}, 80%, 60%)`, flexShrink: 0 }} />
                      <span style={{ fontSize: 13 }}>{f}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div style={{ marginTop: 48, paddingTop: 24, borderTop: "1px solid rgba(0,229,255,0.08)", display: "flex", justifyContent: "space-between", fontSize: 11, color: "#2a4a5a", flexWrap: "wrap", gap: 8 }}>
          <span>Built with Django · PyTorch · React · PatchTST</span>
          <span>Dataset: UCI Household Electric Power Consumption</span>
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes fadeDown { from { opacity: 0; transform: translateY(-16px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #050d14; }
        ::-webkit-scrollbar-thumb { background: rgba(0,229,255,0.2); border-radius: 3px; }
        button:hover { opacity: 0.85; }
      `}</style>
    </div>
  );
}
