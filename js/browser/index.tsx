import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import { useTheme, ThemeColors } from "../theme.ts";

// ============================================================================
// Types
// ============================================================================

interface DatasetEntry {
  name: string;
  technique: string;
  shape: number[];
  dtype: string;
  description: string;
  size_mb: number;
}

interface DatasetInfo {
  name: string;
  technique: string;
  description: string;
  data?: { shape: number[]; dtype: string };
  attribution?: { contributor: string; license: string; institution?: string; date?: string };
  instrument?: { microscope?: string; detector?: string; voltage_kv?: number };
  calibration?: { pixel_size?: number; pixel_size_unit?: string };
}

// ============================================================================
// Style constants (matches quantem.widget conventions)
// ============================================================================

const SP = 4;
const FONT = `-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
const MONO = `'SF Mono', 'Fira Code', Menlo, Consolas, monospace`;

const TECHNIQUE_COLORS: Record<string, string> = {
  "4dstem": "#e8a838",
  hrtem: "#5ab8e0",
  eels: "#8bc34a",
  tomo: "#ce93d8",
  diffraction: "#ff8a65",
  complex: "#4dd0e1",
  image: "#90a4ae",
};

function styles(c: ThemeColors) {
  return {
    root: {
      fontFamily: FONT,
      fontSize: 11,
      color: c.text,
      maxWidth: 680,
    },
    header: {
      display: "flex",
      alignItems: "center",
      gap: SP * 3,
      padding: `${SP + 2}px ${SP * 3}px`,
      borderBottom: `1px solid ${c.border}`,
      background: c.controlBg,
    },
    title: {
      fontWeight: 600,
      fontSize: 12,
      letterSpacing: 0.2,
      whiteSpace: "nowrap" as const,
    },
    filterGroup: {
      display: "flex",
      alignItems: "center",
      gap: SP,
      marginLeft: "auto",
    },
    select: {
      fontFamily: FONT,
      fontSize: 10,
      padding: "2px 6px",
      border: `1px solid ${c.border}`,
      background: c.controlBg,
      color: c.text,
      cursor: "pointer",
      outline: "none",
    },
    btn: {
      fontFamily: FONT,
      fontSize: 10,
      padding: "2px 8px",
      border: `1px solid ${c.border}`,
      background: c.controlBg,
      color: c.textMuted,
      cursor: "pointer",
    },
    tableWrap: {
      maxHeight: 280,
      overflowY: "auto" as const,
    },
    table: {
      width: "100%",
      borderCollapse: "collapse" as const,
      fontSize: 10,
    },
    th: {
      textAlign: "left" as const,
      padding: `${SP}px ${SP * 2}px`,
      borderBottom: `1px solid ${c.border}`,
      background: c.controlBg,
      fontWeight: 600,
      fontSize: 9,
      color: c.textMuted,
      textTransform: "uppercase" as const,
      letterSpacing: 0.6,
      position: "sticky" as const,
      top: 0,
      zIndex: 1,
    },
    td: {
      padding: `${SP}px ${SP * 2}px`,
      borderBottom: `1px solid ${c.border}`,
    },
    mono: {
      fontFamily: MONO,
      fontSize: 10,
    },
    detail: {
      padding: `${SP * 2}px ${SP * 3}px`,
      borderTop: `1px solid ${c.border}`,
      background: c.controlBg,
    },
    detailGrid: {
      display: "grid",
      gridTemplateColumns: "90px 1fr",
      gap: `${SP - 1}px ${SP * 3}px`,
      alignItems: "baseline",
    },
    detailLabel: {
      fontWeight: 600,
      color: c.textMuted,
      fontSize: 9,
      textTransform: "uppercase" as const,
      letterSpacing: 0.4,
    },
    detailValue: {
      fontFamily: MONO,
      fontSize: 10,
    },
    statusBar: {
      padding: `${SP}px ${SP * 3}px`,
      borderTop: `1px solid ${c.border}`,
      fontSize: 10,
      color: c.textMuted,
      display: "flex",
      justifyContent: "space-between",
    },
    loadBtn: {
      fontFamily: FONT,
      fontSize: 10,
      fontWeight: 600,
      padding: "3px 14px",
      border: "none",
      background: c.accent,
      color: "#fff",
      cursor: "pointer",
      letterSpacing: 0.3,
    },
    badge: {
      display: "inline-block",
      padding: "0px 4px",
      fontSize: 9,
      fontWeight: 500,
      letterSpacing: 0.3,
    },
  } as const;
}

// ============================================================================
// Helpers
// ============================================================================

function formatShape(shape: number[]): string {
  if (!shape || shape.length === 0) return "\u2014";
  return shape.join(" \u00d7 ");
}

function formatSize(mb: number): string {
  if (!mb || mb === 0) return "\u2014";
  if (mb < 1) return `${(mb * 1024).toFixed(0)} KB`;
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb.toFixed(1)} MB`;
}

// ============================================================================
// Widget
// ============================================================================

function DataBrowser() {
  const { colors } = useTheme();
  const s = React.useMemo(() => styles(colors), [colors]);

  const [catalogJson] = useModelState<string>("catalog_json");
  const [selectedName, setSelectedName] = useModelState<string>("selected_name");
  const [selectedInfoJson] = useModelState<string>("selected_info_json");
  const [techniqueFilter, setTechniqueFilter] = useModelState<string>("technique_filter");
  const [loading] = useModelState<boolean>("loading");
  const [loadedName] = useModelState<string>("loaded_name");
  const [, setRefreshRequested] = useModelState<boolean>("_refresh_requested");
  const [, setLoadRequested] = useModelState<boolean>("_load_requested");

  const catalog: DatasetEntry[] = React.useMemo(() => {
    try { return JSON.parse(catalogJson || "[]"); }
    catch { return []; }
  }, [catalogJson]);

  const techniques = React.useMemo(() => {
    const set = new Set(catalog.map((d) => d.technique));
    return Array.from(set).sort();
  }, [catalog]);

  const filtered = React.useMemo(() => {
    if (!techniqueFilter) return catalog;
    return catalog.filter((d) => d.technique === techniqueFilter);
  }, [catalog, techniqueFilter]);

  const selectedInfo: DatasetInfo | null = React.useMemo(() => {
    try {
      if (!selectedInfoJson) return null;
      return JSON.parse(selectedInfoJson) as DatasetInfo;
    } catch { return null; }
  }, [selectedInfoJson]);

  const handleRowClick = React.useCallback(
    (name: string) => setSelectedName(name),
    [setSelectedName],
  );

  const handleLoad = React.useCallback(() => {
    if (selectedName) setLoadRequested(true);
  }, [selectedName, setLoadRequested]);

  const handleRefresh = React.useCallback(
    () => setRefreshRequested(true),
    [setRefreshRequested],
  );

  return (
    <div style={s.root}>
      {/* ── Header ── */}
      <div style={s.header}>
        <span style={s.title}>quantem.data</span>
        <div style={s.filterGroup}>
          <select
            style={s.select}
            value={techniqueFilter}
            onChange={(e) => setTechniqueFilter(e.target.value)}
          >
            <option value="">All techniques</option>
            {techniques.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
          <button
            style={{ ...s.btn, opacity: loading ? 0.5 : 1 }}
            onClick={handleRefresh}
            disabled={loading}
            title="Refresh catalog from HF Hub"
          >
            {loading ? "\u2026" : "Refresh"}
          </button>
        </div>
      </div>

      {/* ── Table ── */}
      <div style={s.tableWrap}>
        {catalog.length === 0 ? (
          <div style={{ padding: SP * 6, textAlign: "center", color: colors.textMuted, fontSize: 10 }}>
            {loading ? "Loading catalog\u2026" : "No datasets found"}
          </div>
        ) : (
          <table style={s.table}>
            <thead>
              <tr>
                <th style={s.th}>Name</th>
                <th style={s.th}>Technique</th>
                <th style={s.th}>Shape</th>
                <th style={s.th}>Size</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((d) => {
                const sel = d.name === selectedName;
                const loaded = d.name === loadedName;
                const techColor = TECHNIQUE_COLORS[d.technique] || colors.textMuted;

                return (
                  <tr
                    key={d.name}
                    onClick={() => handleRowClick(d.name)}
                    style={{
                      cursor: "pointer",
                      background: sel ? colors.bgAlt : "transparent",
                      borderLeft: sel ? `2px solid ${colors.accent}` : "2px solid transparent",
                    }}
                    onMouseEnter={(e) => {
                      if (!sel) e.currentTarget.style.background = colors.bgAlt;
                    }}
                    onMouseLeave={(e) => {
                      if (!sel) e.currentTarget.style.background = "transparent";
                    }}
                  >
                    <td style={s.td}>
                      <span style={{ fontWeight: sel ? 600 : 400 }}>{d.name}</span>
                      {loaded && (
                        <span
                          style={{
                            ...s.badge,
                            marginLeft: 6,
                            color: "#4caf50",
                          }}
                        >
                          \u2713
                        </span>
                      )}
                    </td>
                    <td style={s.td}>
                      <span style={{ ...s.badge, color: techColor, border: `1px solid ${techColor}40` }}>
                        {d.technique}
                      </span>
                    </td>
                    <td style={{ ...s.td, ...s.mono }}>{formatShape(d.shape)}</td>
                    <td style={{ ...s.td, ...s.mono, color: colors.textMuted }}>{formatSize(d.size_mb)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* ── Detail panel ── */}
      {selectedName && selectedInfo && (
        <div style={s.detail}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: SP * 2 }}>
            <span style={{ fontWeight: 600, fontSize: 11 }}>{selectedName}</span>
            <button
              style={{ ...s.loadBtn, opacity: loading ? 0.5 : 1, cursor: loading ? "wait" : "pointer" }}
              onClick={handleLoad}
              disabled={loading}
            >
              {loading ? "LOADING\u2026" : loadedName === selectedName ? "RELOAD" : "LOAD"}
            </button>
          </div>
          <div style={s.detailGrid}>
            {selectedInfo.description && (
              <>
                <span style={s.detailLabel}>Description</span>
                <span style={{ fontSize: 10 }}>{selectedInfo.description}</span>
              </>
            )}
            <span style={s.detailLabel}>Technique</span>
            <span style={s.detailValue}>{selectedInfo.technique || "\u2014"}</span>
            {selectedInfo.data && (
              <>
                <span style={s.detailLabel}>Shape</span>
                <span style={s.detailValue}>{formatShape(selectedInfo.data.shape)}</span>
                <span style={s.detailLabel}>Dtype</span>
                <span style={s.detailValue}>{selectedInfo.data.dtype || "\u2014"}</span>
              </>
            )}
            {selectedInfo.attribution && (
              <>
                <span style={s.detailLabel}>Contributor</span>
                <span style={{ fontSize: 10 }}>{selectedInfo.attribution.contributor || "\u2014"}</span>
                <span style={s.detailLabel}>License</span>
                <span style={{ fontSize: 10 }}>{selectedInfo.attribution.license || "\u2014"}</span>
              </>
            )}
            {selectedInfo.instrument?.microscope && (
              <>
                <span style={s.detailLabel}>Microscope</span>
                <span style={{ fontSize: 10 }}>{selectedInfo.instrument.microscope}</span>
              </>
            )}
            {selectedInfo.calibration?.pixel_size != null && (
              <>
                <span style={s.detailLabel}>Pixel size</span>
                <span style={s.detailValue}>
                  {selectedInfo.calibration.pixel_size} {selectedInfo.calibration.pixel_size_unit || ""}
                </span>
              </>
            )}
          </div>
        </div>
      )}

      {/* ── Status bar ── */}
      <div style={s.statusBar}>
        <span>
          {filtered.length} dataset{filtered.length !== 1 ? "s" : ""}
          {techniqueFilter ? ` \u00b7 ${techniqueFilter}` : ""}
        </span>
        {loadedName && (
          <span>
            Loaded: <span style={{ fontFamily: MONO, fontWeight: 600 }}>{loadedName}</span>
          </span>
        )}
      </div>
    </div>
  );
}

export const render = createRender(DataBrowser);
