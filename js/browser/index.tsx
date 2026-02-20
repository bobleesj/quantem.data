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
  attribution?: { contributor: string; license: string };
}

// ============================================================================
// Style constants
// ============================================================================

const SPACING = 6;
const FONT_FAMILY = `-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
const MONO_FAMILY = `'SF Mono', 'Fira Code', 'Cascadia Code', Menlo, Consolas, monospace`;

function makeStyles(colors: ThemeColors) {
  return {
    container: {
      fontFamily: FONT_FAMILY,
      fontSize: 11,
      color: colors.text,
      background: colors.bg,
      border: `1px solid ${colors.border}`,
      maxWidth: 720,
      overflow: "hidden" as const,
    },
    header: {
      display: "flex",
      alignItems: "center",
      gap: SPACING * 2,
      padding: `${SPACING}px ${SPACING * 2}px`,
      borderBottom: `1px solid ${colors.border}`,
      background: colors.controlBg,
    },
    title: {
      fontWeight: 600,
      fontSize: 12,
      whiteSpace: "nowrap" as const,
    },
    filterRow: {
      display: "flex",
      alignItems: "center",
      gap: SPACING,
      marginLeft: "auto",
    },
    select: {
      fontFamily: FONT_FAMILY,
      fontSize: 10,
      padding: "2px 4px",
      border: `1px solid ${colors.border}`,
      background: colors.controlBg,
      color: colors.text,
      cursor: "pointer",
    },
    button: {
      fontFamily: FONT_FAMILY,
      fontSize: 10,
      padding: "2px 8px",
      border: `1px solid ${colors.border}`,
      background: colors.controlBg,
      color: colors.text,
      cursor: "pointer",
    },
    tableWrap: {
      maxHeight: 320,
      overflowY: "auto" as const,
    },
    table: {
      width: "100%",
      borderCollapse: "collapse" as const,
      fontSize: 10,
    },
    th: {
      textAlign: "left" as const,
      padding: `${SPACING}px ${SPACING * 1.5}px`,
      borderBottom: `1px solid ${colors.border}`,
      background: colors.controlBg,
      fontWeight: 600,
      fontSize: 10,
      color: colors.textMuted,
      textTransform: "uppercase" as const,
      letterSpacing: 0.5,
      position: "sticky" as const,
      top: 0,
      zIndex: 1,
    },
    td: {
      padding: `${SPACING - 1}px ${SPACING * 1.5}px`,
      borderBottom: `1px solid ${colors.border}`,
    },
    mono: {
      fontFamily: MONO_FAMILY,
      fontSize: 10,
    },
    detail: {
      padding: `${SPACING * 2}px`,
      borderTop: `1px solid ${colors.border}`,
      background: colors.controlBg,
      fontSize: 10,
    },
    detailGrid: {
      display: "grid",
      gridTemplateColumns: "auto 1fr",
      gap: `${SPACING - 2}px ${SPACING * 2}px`,
      alignItems: "baseline",
    },
    detailLabel: {
      fontWeight: 600,
      color: colors.textMuted,
      fontSize: 10,
      textTransform: "uppercase" as const,
    },
    detailValue: {
      fontFamily: MONO_FAMILY,
      fontSize: 10,
    },
    statusBar: {
      padding: `${SPACING - 2}px ${SPACING * 2}px`,
      borderTop: `1px solid ${colors.border}`,
      fontSize: 10,
      color: colors.textMuted,
      display: "flex",
      justifyContent: "space-between",
    },
    loadButton: {
      fontFamily: FONT_FAMILY,
      fontSize: 10,
      fontWeight: 600,
      padding: "3px 12px",
      border: `1px solid ${colors.accent}`,
      background: colors.accent,
      color: "#fff",
      cursor: "pointer",
    },
    badge: {
      display: "inline-block",
      padding: "1px 5px",
      fontSize: 9,
      fontWeight: 500,
      border: `1px solid ${colors.border}`,
      background: colors.bgAlt,
      color: colors.textMuted,
    },
  } as const;
}

// ============================================================================
// Technique colors (subtle)
// ============================================================================

const TECHNIQUE_COLORS: Record<string, string> = {
  "4dstem": "#e8a838",
  "hrtem": "#5ab8e0",
  "eels": "#8bc34a",
  "tomo": "#ce93d8",
  "diffraction": "#ff8a65",
  "complex": "#4dd0e1",
  "image": "#90a4ae",
};

// ============================================================================
// Widget
// ============================================================================

function DataBrowser() {
  const { colors } = useTheme();
  const styles = React.useMemo(() => makeStyles(colors), [colors]);

  const [catalogJson] = useModelState<string>("catalog_json");
  const [selectedName, setSelectedName] = useModelState<string>("selected_name");
  const [selectedInfoJson] = useModelState<string>("selected_info_json");
  const [techniqueFilter, setTechniqueFilter] = useModelState<string>("technique_filter");
  const [loading] = useModelState<boolean>("loading");
  const [loadedName] = useModelState<string>("loaded_name");
  const [, setRefreshRequested] = useModelState<boolean>("_refresh_requested");
  const [, setLoadRequested] = useModelState<boolean>("_load_requested");

  // Parse catalog
  const catalog: DatasetEntry[] = React.useMemo(() => {
    try {
      return JSON.parse(catalogJson || "[]");
    } catch {
      return [];
    }
  }, [catalogJson]);

  // Available techniques
  const techniques = React.useMemo(() => {
    const set = new Set(catalog.map((d) => d.technique));
    return Array.from(set).sort();
  }, [catalog]);

  // Filter catalog
  const filtered = React.useMemo(() => {
    if (!techniqueFilter) return catalog;
    return catalog.filter((d) => d.technique === techniqueFilter);
  }, [catalog, techniqueFilter]);

  // Parse selected info
  const selectedInfo: DatasetInfo | null = React.useMemo(() => {
    try {
      if (!selectedInfoJson) return null;
      return JSON.parse(selectedInfoJson) as DatasetInfo;
    } catch {
      return null;
    }
  }, [selectedInfoJson]);

  // Row click
  const handleRowClick = React.useCallback(
    (name: string) => {
      setSelectedName(name);
    },
    [setSelectedName]
  );

  // Load button
  const handleLoad = React.useCallback(() => {
    if (selectedName) {
      setLoadRequested(true);
    }
  }, [selectedName, setLoadRequested]);

  // Refresh
  const handleRefresh = React.useCallback(() => {
    setRefreshRequested(true);
  }, [setRefreshRequested]);

  const formatShape = (shape: number[]) => {
    if (!shape || shape.length === 0) return "—";
    return shape.join(" \u00d7 ");
  };

  const formatSize = (mb: number) => {
    if (!mb || mb === 0) return "—";
    if (mb < 1) return `${(mb * 1024).toFixed(0)} KB`;
    if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
    return `${mb.toFixed(1)} MB`;
  };

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <span style={styles.title}>quantem.data</span>
        <div style={styles.filterRow}>
          <select
            style={styles.select}
            value={techniqueFilter}
            onChange={(e) => setTechniqueFilter(e.target.value)}
          >
            <option value="">All techniques</option>
            {techniques.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <button
            style={styles.button}
            onClick={handleRefresh}
            disabled={loading}
            title="Refresh catalog from HF Hub"
          >
            {loading ? "..." : "Refresh"}
          </button>
        </div>
      </div>

      {/* Table */}
      <div style={styles.tableWrap}>
        {catalog.length === 0 ? (
          <div style={{ padding: SPACING * 3, textAlign: "center", color: colors.textMuted }}>
            {loading ? "Loading catalog..." : "No datasets found"}
          </div>
        ) : (
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>Name</th>
                <th style={styles.th}>Technique</th>
                <th style={styles.th}>Shape</th>
                <th style={styles.th}>Size</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((d) => {
                const isSelected = d.name === selectedName;
                const isLoaded = d.name === loadedName;
                const rowBg = isSelected
                  ? colors.bgAlt
                  : "transparent";

                return (
                  <tr
                    key={d.name}
                    onClick={() => handleRowClick(d.name)}
                    style={{
                      cursor: "pointer",
                      background: rowBg,
                    }}
                    onMouseEnter={(e) => {
                      if (!isSelected) e.currentTarget.style.background = colors.bgAlt;
                    }}
                    onMouseLeave={(e) => {
                      if (!isSelected) e.currentTarget.style.background = "transparent";
                    }}
                  >
                    <td style={styles.td}>
                      <span style={{ fontWeight: isSelected ? 600 : 400 }}>{d.name}</span>
                      {isLoaded && (
                        <span
                          style={{
                            ...styles.badge,
                            marginLeft: 6,
                            color: "#4caf50",
                            borderColor: "#4caf50",
                          }}
                        >
                          loaded
                        </span>
                      )}
                    </td>
                    <td style={styles.td}>
                      <span
                        style={{
                          ...styles.badge,
                          borderColor: TECHNIQUE_COLORS[d.technique] || colors.border,
                          color: TECHNIQUE_COLORS[d.technique] || colors.textMuted,
                        }}
                      >
                        {d.technique}
                      </span>
                    </td>
                    <td style={{ ...styles.td, ...styles.mono }}>{formatShape(d.shape)}</td>
                    <td style={{ ...styles.td, ...styles.mono }}>{formatSize(d.size_mb)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Detail panel */}
      {selectedName && selectedInfo && (
        <div style={styles.detail}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: SPACING }}>
            <span style={{ fontWeight: 600, fontSize: 11 }}>{selectedName}</span>
            <button
              style={{
                ...styles.loadButton,
                opacity: loading ? 0.6 : 1,
                cursor: loading ? "wait" : "pointer",
              }}
              onClick={handleLoad}
              disabled={loading}
            >
              {loading ? "Loading..." : loadedName === selectedName ? "Reload" : "Load"}
            </button>
          </div>
          <div style={styles.detailGrid}>
            {selectedInfo.description && (
              <>
                <span style={styles.detailLabel}>Description</span>
                <span>{selectedInfo.description}</span>
              </>
            )}
            <span style={styles.detailLabel}>Technique</span>
            <span>{selectedInfo.technique || "\u2014"}</span>
            {selectedInfo.data && (
              <>
                <span style={styles.detailLabel}>Shape</span>
                <span style={styles.detailValue}>
                  {formatShape(selectedInfo.data.shape)}
                </span>
                <span style={styles.detailLabel}>Dtype</span>
                <span style={styles.detailValue}>
                  {selectedInfo.data.dtype || "\u2014"}
                </span>
              </>
            )}
            {selectedInfo.attribution && (
              <>
                <span style={styles.detailLabel}>Contributor</span>
                <span>{selectedInfo.attribution.contributor || "\u2014"}</span>
                <span style={styles.detailLabel}>License</span>
                <span>{selectedInfo.attribution.license || "\u2014"}</span>
              </>
            )}
          </div>
        </div>
      )}

      {/* Status bar */}
      <div style={styles.statusBar}>
        <span>
          {filtered.length} dataset{filtered.length !== 1 ? "s" : ""}
          {techniqueFilter ? ` (${techniqueFilter})` : ""}
        </span>
        {loadedName && (
          <span>
            Loaded: <span style={{ fontFamily: MONO_FAMILY, fontWeight: 600 }}>{loadedName}</span>
          </span>
        )}
      </div>
    </div>
  );
}

export const render = createRender(DataBrowser);
