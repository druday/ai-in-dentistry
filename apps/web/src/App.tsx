import { useEffect, useMemo, useState } from "react";

type AnyMap = Record<string, any>;

type TemplateResponse = {
  default_template_id: string;
  templates: Array<{
    id: string;
    name: string;
    description: string;
    protocol: AnyMap;
  }>;
};

type RunStatus = {
  job_id: string;
  status: string;
  created_at: string;
  updated_at: string;
  stage: string;
  error?: string;
  result?: AnyMap;
  log_path: string;
};

type ArtifactItem = {
  path: string;
  relative_path: string;
  name: string;
  kind: string;
};

type DescriptiveTableResponse = {
  job_id: string;
  available: boolean;
  csv_path: string;
  md_path: string;
  columns: string[];
  rows: AnyMap[];
  row_count?: number;
};

const api = async <T,>(url: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
};

const deepClone = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

const listToText = (items: string[]) => items.join(", ");
const textToList = (text: string) =>
  text
    .split(/[\n,]/)
    .map((x) => x.trim())
    .filter(Boolean);

const ensureQueryDateFields = (protocol: AnyMap): AnyMap => {
  const copy = deepClone(protocol);
  copy.queries = (copy.queries ?? []).map((query: AnyMap) => ({
    ...query,
    start_date: query.start_date ?? "",
    end_date: query.end_date ?? ""
  }));
  return copy;
};

const prepareProtocolForRun = (protocol: AnyMap): AnyMap => {
  const copy = deepClone(protocol);
  copy.queries = (copy.queries ?? []).map((query: AnyMap) => {
    const start = String(query.start_date ?? "").trim();
    const end = String(query.end_date ?? "").trim();
    const rawQuery = String(query.query ?? "").trim();
    if (!start || !end) {
      const { start_date: _s, end_date: _e, ...rest } = query;
      return rest;
    }
    const dateBlock = `(("${start}"[Date - Publication] : "${end}"[Date - Publication]))`;
    const merged = rawQuery.includes("[Date - Publication]")
      ? rawQuery
      : `${rawQuery} AND ${dateBlock}`;
    const { start_date: _s, end_date: _e, ...rest } = query;
    return {
      ...rest,
      query: merged
    };
  });
  return copy;
};

export default function App() {
  const [templates, setTemplates] = useState<TemplateResponse | null>(null);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>("");
  const [protocol, setProtocol] = useState<AnyMap | null>(null);
  const [validationMessage, setValidationMessage] = useState<string>("");
  const [previewRows, setPreviewRows] = useState<Array<AnyMap>>([]);
  const [previewError, setPreviewError] = useState<string>("");
  const [runJobId, setRunJobId] = useState<string>("");
  const [runStatus, setRunStatus] = useState<RunStatus | null>(null);
  const [runLog, setRunLog] = useState<string>("");
  const [artifacts, setArtifacts] = useState<string[]>([]);
  const [artifactItems, setArtifactItems] = useState<ArtifactItem[]>([]);
  const [descriptiveTable, setDescriptiveTable] = useState<DescriptiveTableResponse | null>(null);
  const [busy, setBusy] = useState<boolean>(false);
  const [runOptions, setRunOptions] = useState({
    fetch_first: false,
    clean_output: false,
    period_mode: "dynamic",
    with_funding: true,
    generate_descriptive_table: true,
    generate_flow_figures: true,
    generate_sankey_html: true,
    raw_glob: "data/raw/pubmed_records_*.jsonl",
    fetch_output_dir: "data/raw"
  });

  useEffect(() => {
    void (async () => {
      const response = await api<TemplateResponse>("/api/templates");
      setTemplates(response);
      const defaultId = response.default_template_id || response.templates[0]?.id || "";
      setSelectedTemplateId(defaultId);
      const selected = response.templates.find((x) => x.id === defaultId);
      if (selected) {
        setProtocol(ensureQueryDateFields(selected.protocol));
      }
    })();
  }, []);

  useEffect(() => {
    if (!runJobId) {
      return;
    }
    const tick = async () => {
      const status = await api<RunStatus>(`/api/run/${runJobId}`);
      setRunStatus(status);
      const logs = await api<{ log: string }>(`/api/run/${runJobId}/logs`);
      setRunLog(logs.log);
      if (status.status === "completed" || status.status === "failed") {
        const artifactResponse = await api<{ artifacts: string[]; artifact_items?: ArtifactItem[] }>(
          `/api/run/${runJobId}/artifacts`
        );
        setArtifacts(artifactResponse.artifacts ?? []);
        setArtifactItems(artifactResponse.artifact_items ?? []);
        const tableResponse = await api<DescriptiveTableResponse>(
          `/api/run/${runJobId}/descriptive-table`
        );
        setDescriptiveTable(tableResponse);
      }
    };
    void tick();
    const interval = window.setInterval(() => {
      void tick();
    }, 2500);
    return () => window.clearInterval(interval);
  }, [runJobId]);

  const selectedTemplate = useMemo(
    () => templates?.templates.find((x) => x.id === selectedTemplateId),
    [selectedTemplateId, templates]
  );

  const setDynamic = (patch: AnyMap) => {
    setProtocol((current) => {
      if (!current) {
        return current;
      }
      return {
        ...current,
        temporal_segmentation: {
          ...(current.temporal_segmentation ?? {}),
          dynamic: {
            ...(current.temporal_segmentation?.dynamic ?? {}),
            ...patch
          }
        }
      };
    });
  };

  const setClassificationList = (key: "dental" | "medical" | "technical", value: string) => {
    setProtocol((current) => {
      if (!current) {
        return current;
      }
      return {
        ...current,
        classification: {
          ...(current.classification ?? {}),
          [key]: textToList(value)
        }
      };
    });
  };

  const setFundingList = (value: string) => {
    setProtocol((current) => {
      if (!current) {
        return current;
      }
      return {
        ...current,
        funding: {
          ...(current.funding ?? {}),
          us_federal_keywords: textToList(value)
        }
      };
    });
  };

  const validateProtocol = async () => {
    if (!protocol) {
      return;
    }
    try {
      setBusy(true);
      const response = await api<{ valid: boolean; error?: string }>("/api/protocol/validate", {
        method: "POST",
        body: JSON.stringify({ protocol: prepareProtocolForRun(protocol) })
      });
      setValidationMessage(response.valid ? "Protocol is valid." : `Validation error: ${response.error}`);
    } catch (error) {
      setValidationMessage(`Validation failed: ${(error as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const previewDynamicBins = async () => {
    if (!protocol?.temporal_segmentation?.dynamic) {
      return;
    }
    try {
      setBusy(true);
      setPreviewError("");
      const dynamic = protocol.temporal_segmentation.dynamic;
      const response = await api<{ periods: AnyMap[] }>("/api/periods/preview", {
        method: "POST",
        body: JSON.stringify({
          raw_glob: runOptions.raw_glob,
          n_bins: Number(dynamic.n_bins ?? 6),
          min_year: Number(dynamic.min_year ?? 1946),
          max_year: Number(dynamic.max_year ?? 2025),
          min_years_per_bin: Number(dynamic.min_years_per_bin ?? 1),
          use_observed_year_range: Boolean(dynamic.use_observed_year_range ?? true)
        })
      });
      setPreviewRows(response.periods ?? []);
    } catch (error) {
      setPreviewError((error as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const startRun = async () => {
    if (!protocol) {
      return;
    }
    try {
      setBusy(true);
      const payload = {
        protocol: prepareProtocolForRun(protocol),
        protocol_path: "config/protocol.yaml",
        raw_glob: runOptions.raw_glob,
        fetch_first: runOptions.fetch_first,
        fetch_output_dir: runOptions.fetch_output_dir,
        clean_output: runOptions.clean_output,
        period_mode: runOptions.period_mode,
        with_funding: runOptions.with_funding,
        generate_descriptive_table: runOptions.generate_descriptive_table,
        generate_flow_figures: runOptions.generate_flow_figures,
        generate_sankey_html: runOptions.generate_sankey_html
      };
      const response = await api<{ job_id: string }>("/api/run", {
        method: "POST",
        body: JSON.stringify(payload)
      });
      setRunJobId(response.job_id);
      setRunStatus(null);
      setRunLog("");
      setArtifacts([]);
      setArtifactItems([]);
      setDescriptiveTable(null);
    } catch (error) {
      setValidationMessage(`Run start failed: ${(error as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  if (!templates || !protocol) {
    return <div className="loading">Loading reproducibility console...</div>;
  }

  const resourceHref = (path: string) => `/api/resource?path=${encodeURIComponent(path)}`;
  const hasDescriptiveTable = Boolean(descriptiveTable?.available && descriptiveTable.columns.length > 0);

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="kicker">Reproducibility Console</p>
          <h1>Institutional Network Pipeline Control Center</h1>
          <p className="subtitle">
            Configure search strategy, period binning, institution typing, and funding-stratified
            analysis in one run-ready interface.
          </p>
        </div>
        <div className="templateBox">
          <label>Domain preset</label>
          <select
            value={selectedTemplateId}
            onChange={(event) => {
              const id = event.target.value;
              setSelectedTemplateId(id);
              const selected = templates.templates.find((x) => x.id === id);
              if (selected) {
                setProtocol(ensureQueryDateFields(selected.protocol));
              }
            }}
          >
            {templates.templates.map((template) => (
              <option key={template.id} value={template.id}>
                {template.name}
              </option>
            ))}
          </select>
          <p>{selectedTemplate?.description}</p>
        </div>
      </header>

      <section className="card">
        <div className="cardHeader">
          <h2>Search strategy</h2>
          <button
            className="ghost"
            onClick={() =>
              setProtocol((current) => ({
                ...current!,
                queries: [
                  ...(current?.queries ?? []),
                  {
                    id: `custom_query_${(current?.queries ?? []).length + 1}`,
                    query: "",
                    start_date: "2020/01/01",
                    end_date: "2025/12/31"
                  }
                ]
              }))
            }
          >
            Add more
          </button>
        </div>
        {(protocol.queries ?? []).map((query: AnyMap, idx: number) => (
          <div key={`${query.id}-${idx}`} className="queryRow">
            <input
              value={query.id ?? ""}
              onChange={(event) =>
                setProtocol((current) => {
                  if (!current) {
                    return current;
                  }
                  const next = deepClone(current);
                  next.queries[idx].id = event.target.value;
                  return next;
                })
              }
              placeholder="query id"
            />
            <input
              value={query.start_date ?? ""}
              onChange={(event) =>
                setProtocol((current) => {
                  if (!current) {
                    return current;
                  }
                  const next = deepClone(current);
                  next.queries[idx].start_date = event.target.value;
                  return next;
                })
              }
              placeholder="start YYYY/MM/DD"
            />
            <input
              value={query.end_date ?? ""}
              onChange={(event) =>
                setProtocol((current) => {
                  if (!current) {
                    return current;
                  }
                  const next = deepClone(current);
                  next.queries[idx].end_date = event.target.value;
                  return next;
                })
              }
              placeholder="end YYYY/MM/DD"
            />
            <button
              className="danger"
              onClick={() =>
                setProtocol((current) => {
                  if (!current) {
                    return current;
                  }
                  const next = deepClone(current);
                  next.queries.splice(idx, 1);
                  return next;
                })
              }
            >
              Remove
            </button>
            <textarea
              value={query.query ?? ""}
              onChange={(event) =>
                setProtocol((current) => {
                  if (!current) {
                    return current;
                  }
                  const next = deepClone(current);
                  next.queries[idx].query = event.target.value;
                  return next;
                })
              }
            />
          </div>
        ))}
      </section>

      <section className="grid">
        <div className="card">
          <h2>Temporal binning</h2>
          <label>Segmentation mode</label>
          <select
            value={runOptions.period_mode}
            onChange={(event) =>
              setRunOptions((current) => ({ ...current, period_mode: event.target.value }))
            }
          >
            <option value="dynamic">Dynamic balanced (default)</option>
            <option value="fixed">Fixed protocol bins</option>
            <option value="balanced">Balanced alias</option>
          </select>

          <div className="inlineGrid">
            <label>n_bins</label>
            <input
              type="number"
              value={protocol.temporal_segmentation?.dynamic?.n_bins ?? 6}
              onChange={(event) => setDynamic({ n_bins: Number(event.target.value) })}
            />
            <label>min_year</label>
            <input
              type="number"
              value={protocol.temporal_segmentation?.dynamic?.min_year ?? 1946}
              onChange={(event) => setDynamic({ min_year: Number(event.target.value) })}
            />
            <label>max_year</label>
            <input
              type="number"
              value={protocol.temporal_segmentation?.dynamic?.max_year ?? 2025}
              onChange={(event) => setDynamic({ max_year: Number(event.target.value) })}
            />
            <label>min_years_per_bin</label>
            <input
              type="number"
              value={protocol.temporal_segmentation?.dynamic?.min_years_per_bin ?? 1}
              onChange={(event) => setDynamic({ min_years_per_bin: Number(event.target.value) })}
            />
          </div>
          <button onClick={previewDynamicBins} disabled={busy}>
            Preview balanced bins by publication count
          </button>
          {previewError ? <p className="error">{previewError}</p> : null}
          {previewRows.length > 0 ? (
            <table>
              <thead>
                <tr>
                  <th>Period</th>
                  <th>Count</th>
                </tr>
              </thead>
              <tbody>
                {previewRows.map((row, index) => (
                  <tr key={`${row.label}-${index}`}>
                    <td>{row.label}</td>
                    <td>{row.publication_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </div>

        <div className="card">
          <h2>Institution type keywords</h2>
          <label>Dental</label>
          <textarea
            value={listToText(protocol.classification?.dental ?? [])}
            onChange={(event) => setClassificationList("dental", event.target.value)}
          />
          <label>Medical</label>
          <textarea
            value={listToText(protocol.classification?.medical ?? [])}
            onChange={(event) => setClassificationList("medical", event.target.value)}
          />
          <label>Technical</label>
          <textarea
            value={listToText(protocol.classification?.technical ?? [])}
            onChange={(event) => setClassificationList("technical", event.target.value)}
          />
        </div>
      </section>

      <section className="grid">
        <div className="card">
          <h2>Preprocessing controls</h2>
          <label>Institution extraction mode</label>
          <select
            value={protocol.preprocessing?.institution_extraction_mode ?? "exact_affiliation"}
            onChange={(event) =>
              setProtocol((current) => ({
                ...current!,
                preprocessing: {
                  ...(current?.preprocessing ?? {}),
                  institution_extraction_mode: event.target.value
                }
              }))
            }
          >
            <option value="exact_affiliation">exact_affiliation</option>
            <option value="keyword_hint_segment">keyword_hint_segment</option>
          </select>
          <label>
            <input
              type="checkbox"
              checked={Boolean(protocol.preprocessing?.deduplicate_within_publication ?? true)}
              onChange={(event) =>
                setProtocol((current) => ({
                  ...current!,
                  preprocessing: {
                    ...(current?.preprocessing ?? {}),
                    deduplicate_within_publication: event.target.checked
                  }
                }))
              }
            />
            Deduplicate institution-country pairs within publication
          </label>
        </div>

        <div className="card">
          <h2>Funding controls</h2>
          <label>
            <input
              type="checkbox"
              checked={Boolean(protocol.funding?.enabled ?? true)}
              onChange={(event) =>
                setProtocol((current) => ({
                  ...current!,
                  funding: {
                    ...(current?.funding ?? {}),
                    enabled: event.target.checked
                  }
                }))
              }
            />
            Enable funding-stratified analysis
          </label>
          <label>US federal whitelist keywords</label>
          <textarea
            value={listToText(protocol.funding?.us_federal_keywords ?? [])}
            onChange={(event) => setFundingList(event.target.value)}
          />
          <p className="small">
            Hierarchy: any US federal signal → US_FEDERAL_FUNDED; else any grant support →
            NON_US_OR_INTL_FUNDED; else NO_GRANT_LISTED.
          </p>
        </div>
      </section>

      <section className="card">
        <h2>Run controls</h2>
        <div className="runOptions">
          <label>
            <input
              type="checkbox"
              checked={runOptions.fetch_first}
              onChange={(event) =>
                setRunOptions((current) => ({ ...current, fetch_first: event.target.checked }))
              }
            />
            Fetch snapshots before analysis
          </label>
          <label>
            <input
              type="checkbox"
              checked={runOptions.clean_output}
              onChange={(event) =>
                setRunOptions((current) => ({ ...current, clean_output: event.target.checked }))
              }
            />
            Clean output directories before run
          </label>
          <label>
            <input
              type="checkbox"
              checked={runOptions.with_funding}
              onChange={(event) =>
                setRunOptions((current) => ({ ...current, with_funding: event.target.checked }))
              }
            />
            Compute funding-stratified outputs
          </label>
          <label>
            <input
              type="checkbox"
              checked={runOptions.generate_descriptive_table}
              onChange={(event) =>
                setRunOptions((current) => ({
                  ...current,
                  generate_descriptive_table: event.target.checked
                }))
              }
            />
            Generate master descriptive table
          </label>
          <label>
            <input
              type="checkbox"
              checked={runOptions.generate_flow_figures}
              onChange={(event) =>
                setRunOptions((current) => ({
                  ...current,
                  generate_flow_figures: event.target.checked
                }))
              }
            />
            Generate role flow PNG/PDF figures
          </label>
          <label>
            <input
              type="checkbox"
              checked={runOptions.generate_sankey_html}
              onChange={(event) =>
                setRunOptions((current) => ({
                  ...current,
                  generate_sankey_html: event.target.checked
                }))
              }
            />
            Generate interactive Sankey HTML files
          </label>
        </div>
        <div className="inlineGrid">
          <label>Raw file glob</label>
          <input
            value={runOptions.raw_glob}
            onChange={(event) =>
              setRunOptions((current) => ({ ...current, raw_glob: event.target.value }))
            }
          />
          <label>Fetch output directory</label>
          <input
            value={runOptions.fetch_output_dir}
            onChange={(event) =>
              setRunOptions((current) => ({ ...current, fetch_output_dir: event.target.value }))
            }
          />
        </div>
        <div className="actions">
          <button onClick={validateProtocol} disabled={busy}>
            Validate protocol
          </button>
          <button onClick={startRun} disabled={busy}>
            Run pipeline
          </button>
        </div>
        {validationMessage ? <p className="small">{validationMessage}</p> : null}
      </section>

      <section className="grid">
        <div className="card">
          <h2>Run status</h2>
          {runStatus ? (
            <>
              <p>Job: {runStatus.job_id}</p>
              <p>Status: {runStatus.status}</p>
              <p>Stage: {runStatus.stage}</p>
              {runStatus.error ? <p className="error">{runStatus.error}</p> : null}
            </>
          ) : (
            <p>No active run.</p>
          )}
        </div>
        <div className="card">
          <h2>Artifacts</h2>
          {artifactItems.length === 0 && artifacts.length === 0 ? (
            <p>No artifacts listed yet.</p>
          ) : (
            <ul className="artifactList">
              {(artifactItems.length > 0 ? artifactItems : artifacts.map((path) => ({
                path,
                relative_path: path,
                name: path.split("/").slice(-1)[0] ?? path,
                kind: "artifact"
              }))).map((item) => (
                <li key={item.path}>
                  <a href={resourceHref(item.path)} target="_blank" rel="noreferrer">
                    {item.relative_path}
                  </a>
                  <span className="tag">{item.kind}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>

      <section className="card">
        <h2>Master descriptive table (all panels)</h2>
        {!hasDescriptiveTable ? (
          <p>No compiled table available yet. Run the pipeline with table generation enabled.</p>
        ) : (
          <>
            <div className="actions">
              <a
                className="linkButton"
                href={resourceHref(descriptiveTable!.csv_path)}
                target="_blank"
                rel="noreferrer"
              >
                Open CSV
              </a>
              <a
                className="linkButton"
                href={resourceHref(descriptiveTable!.md_path)}
                target="_blank"
                rel="noreferrer"
              >
                Open Markdown
              </a>
            </div>
            <p className="small">
              Rows: {descriptiveTable?.row_count ?? descriptiveTable?.rows.length ?? 0}
            </p>
            <div className="tableScroll">
              <table>
                <thead>
                  <tr>
                    {(descriptiveTable?.columns ?? []).map((column) => (
                      <th key={column}>{column}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(descriptiveTable?.rows ?? []).map((row, index) => (
                    <tr key={`table-row-${index}`}>
                      {(descriptiveTable?.columns ?? []).map((column) => (
                        <td key={`${index}-${column}`}>{row[column] ?? ""}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>

      <section className="card">
        <h2>Live logs</h2>
        <pre>{runLog || "Logs will appear after starting a run."}</pre>
      </section>
    </div>
  );
}
