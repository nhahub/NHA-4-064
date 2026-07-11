import  { useCallback, useMemo, useRef, useState } from "react";
import {
  UploadCloud,
  FileJson,
  FileSpreadsheet,
  ShieldAlert,
  ShieldCheck,
  ShieldQuestion,
  ChevronDown,
  RotateCcw,
  Radar,
  Activity,
  Target,
  X,
} from "lucide-react";


function resolveApiBaseUrl() {
  try {
    if (typeof process !== "undefined" && process.env && process.env.REACT_APP_API_URL) {
      return process.env.REACT_APP_API_URL;
    }
  } catch (e) {
    /* process isn't defined in this bundler — ignore and fall through */
  }
  if (typeof window !== "undefined" && window.__FRAUD_API_URL__) {
    return window.__FRAUD_API_URL__;
  }
  return "http://127.0.0.1:8000"; 
}

const API_BASE_URL = resolveApiBaseUrl();

const REQUIRED_FIELDS = [
  "amount",
  "hour",
  "day_of_week",
  "month",
  "is_night",
  "client_mean_amount",
  "amount_to_credit_ratio",
  "tx_count_same_day",
  "client_merchant_freq",
  "is_online",
  "is_chip",
  "has_error",
];

const RISK_META = {
  high: { label: "High risk", color: "#FF5C5C", Icon: ShieldAlert },
  medium: { label: "Medium risk", color: "#FFB020", Icon: ShieldQuestion },
  low: { label: "Low risk", color: "#30D5A6", Icon: ShieldCheck },
};


function RiskRing({ probability, color, size = 46 }) {
  const stroke = 4;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const offset = c * (1 - probability);
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="fd-ring">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#26303F" strokeWidth={stroke} />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeDasharray={c}
        strokeDashoffset={offset}
        strokeLinecap="round"
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
        style={{ transition: "stroke-dashoffset 0.6s ease" }}
      />
      <text x="50%" y="52%" textAnchor="middle" dominantBaseline="middle" className="fd-ring-label">
        {Math.round(probability * 100)}
      </text>
    </svg>
  );
}

function StatCard({ label, value, sub, accent }) {
  return (
    <div className="fd-stat">
      <div className="fd-stat-label">{label}</div>
      <div className="fd-stat-value" style={accent ? { color: accent } : undefined}>
        {value}
      </div>
      {sub && <div className="fd-stat-sub">{sub}</div>}
    </div>
  );
}

function TransactionRow({ row, raw }) {
  const [open, setOpen] = useState(false);
  const meta = RISK_META[row.risk_level] || RISK_META.low;
  const mismatch = row.true_label !== null && row.true_label !== undefined && row.true_label !== row.prediction;

  return (
    <div className={`fd-row ${open ? "fd-row-open" : ""}`}>
      <button className="fd-row-head" onClick={() => setOpen((o) => !o)}>
        <span className="fd-row-index">#{String(row.index + 1).padStart(3, "0")}</span>
        <RiskRing probability={row.fraud_probability} color={meta.color} />
        <span className="fd-row-badge" style={{ color: meta.color, borderColor: meta.color + "55" }}>
          <meta.Icon size={14} />
          {meta.label}
        </span>
        <span className="fd-row-verdict">{row.prediction === 1 ? "Flagged as fraud" : "Looks legitimate"}</span>
        {mismatch && <span className="fd-row-mismatch">ground truth: {row.true_label === 1 ? "fraud" : "legit"}</span>}
        <span className="fd-row-amount">
          {raw?.amount !== undefined ? `$${Number(raw.amount).toLocaleString(undefined, { maximumFractionDigits: 2 })}` : ""}
        </span>
        <ChevronDown size={16} className="fd-chevron" />
      </button>
      {open && (
        <div className="fd-row-body">
          {REQUIRED_FIELDS.map((f) => (
            <div key={f} className="fd-field">
              <span className="fd-field-key">{f}</span>
              <span className="fd-field-val">{raw && raw[f] !== undefined ? String(raw[f]) : "—"}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


export default function Home() {
  const [rawRows, setRawRows] = useState(null); 
  const [fileMeta, setFileMeta] = useState(null); 
  const [status, setStatus] = useState("idle"); 
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  const parseFile = useCallback(async (file) => {
    setError(null);
    setResult(null);
    setStatus("idle");
    const text = await file.text();
    try {
      let rows;
      if (file.name.toLowerCase().endsWith(".json")) {
        const parsed = JSON.parse(text);
        rows = Array.isArray(parsed) ? parsed : [parsed];
      } else if (file.name.toLowerCase().endsWith(".csv")) {
        const lines = text.trim().split(/\r?\n/);
        const headers = lines[0].split(",").map((h) => h.trim());
        rows = lines.slice(1).map((line) => {
          const cells = line.split(",");
          const obj = {};
          headers.forEach((h, i) => {
            const v = cells[i];
            obj[h] = v !== undefined && v !== "" && !isNaN(Number(v)) ? Number(v) : v;
          });
          return obj;
        });
      } else {
        throw new Error("Only .json and .csv files are supported.");
      }
      if (!rows.length) throw new Error("The file has no rows.");
      setRawRows(rows);
      setFileMeta({ name: file.name, size: file.size, rowCount: rows.length });
    } catch (e) {
      setError(e.message || "Could not parse that file.");
      setRawRows(null);
      setFileMeta(null);
    }
  }, []);

  const onFileChosen = useCallback(
    (fileList) => {
      const file = fileList?.[0];
      if (file) parseFile(file);
    },
    [parseFile]
  );

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragActive(false);
      onFileChosen(e.dataTransfer.files);
    },
    [onFileChosen]
  );

  const runScan = useCallback(async () => {
    if (!rawRows) return;
    setStatus("scanning");
    setError(null);
    try {
      const formData = new FormData();
      const blob = new Blob([JSON.stringify(rawRows)], { type: "application/json" });
      formData.append("file", blob, "data.json");

      const res = await fetch(`${API_BASE_URL}/predict/file`, {
        method: "POST",
        body: formData,
      });
      const body = await res.json();
      if (!res.ok) {
        throw new Error(body.detail ? JSON.stringify(body.detail) : "The API rejected this data.");
      }
      setResult(body);
      setStatus("done");
    } catch (e) {
      setError(e.message || "Could not reach the fraud detection API.");
      setStatus("error");
    }
  }, [rawRows]);

  const reset = useCallback(() => {
    setRawRows(null);
    setFileMeta(null);
    setResult(null);
    setError(null);
    setStatus("idle");
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  const sortedResults = useMemo(() => {
    if (!result) return [];
    return [...result.results].sort((a, b) => b.fraud_probability - a.fraud_probability);
  }, [result]);

  return (
    <div className="fd-root">
      <header className="fd-header">
        <div className="fd-brand">
          <Radar size={36} className="fd-brand-icon" />
          <span>
            RISK<span className="fd-brand-accent">/SCAN</span>
          </span>
        </div>
        <div className="fd-header-sub">AI-powered financial fraud detection</div>
      </header>

      {!result && (
        <section
          className={`fd-dropzone ${dragActive ? "fd-dropzone-active" : ""} ${status === "scanning" ? "fd-dropzone-scanning" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={onDrop}
        >
          <span className="fd-corner fd-corner-tl" />
          <span className="fd-corner fd-corner-tr" />
          <span className="fd-corner fd-corner-bl" />
          <span className="fd-corner fd-corner-br" />
          {status === "scanning" && <span className="fd-scanline" />}

          {!fileMeta ? (
            <>
              <UploadCloud size={34} className="fd-drop-icon" />
              <h2>Drop a batch of transactions to scan</h2>
              <p>Accepts .json or .csv — each row needs the 12 model features (amount, hour, is_night, …).</p>
              <label className="fd-browse-btn">
                Choose file
                <input
                  ref={inputRef}
                  type="file"
                  accept=".json,.csv"
                  hidden
                  onChange={(e) => onFileChosen(e.target.files)}
                />
              </label>
            </>
          ) : (
            <>
              {fileMeta.name.endsWith(".csv") ? (
                <FileSpreadsheet size={30} className="fd-drop-icon" />
              ) : (
                <FileJson size={30} className="fd-drop-icon" />
              )}
              <h2>{fileMeta.name}</h2>
              <p>{fileMeta.rowCount} transaction{fileMeta.rowCount === 1 ? "" : "s"} ready to score</p>
              <div className="fd-actions">
                <button className="fd-primary-btn" onClick={runScan} disabled={status === "scanning"}>
                  <Activity size={16} />
                  {status === "scanning" ? "Scanning…" : "Run fraud scan"}
                </button>
                <button className="fd-ghost-btn" onClick={reset}>
                  <X size={14} /> Clear
                </button>
              </div>
            </>
          )}
        </section>
      )}

      {error && (
        <div className="fd-error">
          <ShieldAlert size={16} />
          <span>{error}</span>
        </div>
      )}

      {result && (
        <section className="fd-results">
          <div className="fd-stat-strip">
            <StatCard label="Transactions scanned" value={result.count} />
            <StatCard
              label="Flagged as fraud"
              value={result.fraud_flagged}
              sub={`${((result.fraud_flagged / result.count) * 100).toFixed(1)}% of batch`}
              accent="#FF5C5C"
            />
            <StatCard
              label="Average risk score"
              value={`${(result.average_probability * 100).toFixed(1)}%`}
              accent="#4C8DFF"
            />
            {result.accuracy !== null && result.accuracy !== undefined && (
              <StatCard
                label="Accuracy vs. ground truth"
                value={`${(result.accuracy * 100).toFixed(1)}%`}
                sub="labels were present in the file"
                accent="#30D5A6"
              />
            )}
          </div>

          <div className="fd-results-head">
            <span>
              <Target size={14} /> Results, ranked by risk
            </span>
            <button className="fd-ghost-btn" onClick={reset}>
              <RotateCcw size={14} /> Scan another file
            </button>
          </div>

          <div className="fd-table">
            {sortedResults.map((row) => (
              <TransactionRow key={row.index} row={row} raw={rawRows?.[row.index]} />
            ))}
          </div>
        </section>
      )}

    </div>
  );
}

