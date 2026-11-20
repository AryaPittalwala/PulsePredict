const { useState, useEffect, useRef, useCallback } = React;

// ── Tiny fetch helper ──────────────────────────────────────────────────────
async function api(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
    return res.json();
}

// ── Mini SVG Sparkline ─────────────────────────────────────────────────────
function Sparkline({ data = [], positive = true, width = 100, height = 36 }) {
    if (!data || data.length < 2) return null;

    const mn = Math.min(...data);
    const mx = Math.max(...data);
    const rng = mx - mn || 1;
    const pad = 3;

    const pts = data
        .map((v, i) => {
            const x = pad + (i / (data.length - 1)) * (width - pad * 2);
            const y = height - pad - ((v - mn) / rng) * (height - pad * 2);
            return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ");

    const color = positive ? "#22c55e" : "#ef4444";
    const lastPt = pts.split(" ").pop().split(",");

    return /*#__PURE__*/ React.createElement(
        "svg",
        { width: width, height: height, viewBox: `0 0 ${width} ${height}` },
    /*#__PURE__*/ React.createElement("polyline", {
            points: pts,
            fill: "none",
            stroke: color,
            strokeWidth: "1.5",
            strokeLinejoin: "round",
            strokeLinecap: "round",
        }),
    /*#__PURE__*/ React.createElement("circle", {
            cx: lastPt[0],
            cy: lastPt[1],
            r: "2.5",
            fill: color,
        })
    );
}

// ── Market Status Pill ─────────────────────────────────────────────────────
function MarketPill({ status }) {
    const isOpen = status?.is_open;
    return /*#__PURE__*/ React.createElement(
        "div",
        { className: "market-pill" },
    /*#__PURE__*/ React.createElement("span", {
            className: "market-pill-dot",
            style: { background: isOpen ? "#22c55e" : "#71717a" },
        }),
    /*#__PURE__*/ React.createElement("span", null, "NSE ", status?.status || "—"),
    /*#__PURE__*/ React.createElement("span", { style: { color: "#52525b" } }, "·"),
    /*#__PURE__*/ React.createElement("span", null, status?.time || "...")
    );
}

// ── Navbar ─────────────────────────────────────────────────────────────────
function Nav({ status, onNavigate }) {
    return /*#__PURE__*/ React.createElement(
        "nav",
        { className: "nav" },
    /*#__PURE__*/ React.createElement(
            "div",
            { className: "nav-logo", onClick: () => onNavigate("landing") },
      /*#__PURE__*/ React.createElement("span", { className: "nav-logo-dot" }),
      /*#__PURE__*/ React.createElement("span", null, "PulsePredict")
        ),
    /*#__PURE__*/ React.createElement(
            "div",
            { className: "nav-links" },
      /*#__PURE__*/ React.createElement(MarketPill, { status: status }),
      /*#__PURE__*/ React.createElement(
                "button",
                { className: "nav-link", onClick: () => onNavigate("predict") },
                "Predict"
            ),
      /*#__PURE__*/ React.createElement(
                "button",
                { className: "nav-link", onClick: () => onNavigate("scoreboard") },
                "Scoreboard"
            ),
      /*#__PURE__*/ React.createElement(
                "button",
                { className: "nav-cta", onClick: () => onNavigate("predict") },
                "Try it free →"
            )
        )
    );
}

// ── Ticker Tape ────────────────────────────────────────────────────────────
function TickerTape({ movers, onSelect }) {
    const items = [...(movers?.gainers || []), ...(movers?.losers || [])];
    const snapshot = items.slice(0, 8);

    return /*#__PURE__*/ React.createElement(
        "div",
        { className: "ticker-tape", style: { marginTop: 84 } },
    /*#__PURE__*/ React.createElement("span", { className: "ticker-label" }, "Market snapshot"),
    /*#__PURE__*/ React.createElement(
            "div",
            { className: "ticker-viewport" },
            items.length
                ? /*#__PURE__*/ React.createElement(
                    "div",
                    { className: "ticker-inner" },
            /*#__PURE__*/ React.createElement(
                        "div",
                        { className: "ticker-group" },
                        snapshot.map((s, index) =>
                /*#__PURE__*/ React.createElement(
                            "button",
                            {
                                key: `${s.ticker}-${index}`,
                                type: "button",
                                className: "ticker-item",
                                onClick: () => onSelect(s.ticker),
                            },
                  /*#__PURE__*/ React.createElement(
                                "span",
                                { className: "ticker-name" },
                                s.ticker?.replace(".NS", "")
                            ),
                  /*#__PURE__*/ React.createElement(
                                "span",
                                { className: "ticker-price" },
                                "₹",
                                s.price?.toLocaleString("en-IN")
                            ),
                  /*#__PURE__*/ React.createElement(
                                "span",
                                { className: s.is_positive ? "ticker-change-pos" : "ticker-change-neg" },
                                s.is_positive ? "▲" : "▼",
                                " ",
                                Math.abs(s.change_pct).toFixed(2),
                                "%"
                            )
                        )
                        )
                    ),
            /*#__PURE__*/ React.createElement(
                        "div",
                        { className: "ticker-group", "aria-hidden": "true" },
                        snapshot.map((s, index) =>
                /*#__PURE__*/ React.createElement(
                            "button",
                            {
                                key: `${s.ticker}-loop-${index}`,
                                type: "button",
                                className: "ticker-item",
                                onClick: () => onSelect(s.ticker),
                            },
                  /*#__PURE__*/ React.createElement(
                                "span",
                                { className: "ticker-name" },
                                s.ticker?.replace(".NS", "")
                            ),
                  /*#__PURE__*/ React.createElement(
                                "span",
                                { className: "ticker-price" },
                                "₹",
                                s.price?.toLocaleString("en-IN")
                            ),
                  /*#__PURE__*/ React.createElement(
                                "span",
                                { className: s.is_positive ? "ticker-change-pos" : "ticker-change-neg" },
                                s.is_positive ? "▲" : "▼",
                                " ",
                                Math.abs(s.change_pct).toFixed(2),
                                "%"
                            )
                        )
                        )
                    )
                )
                : /*#__PURE__*/ React.createElement(
                    "span",
                    { className: "ticker-loading" },
                    "Loading latest market prices..."
                )
        )
    );
}

// ── Stock Row ──────────────────────────────────────────────────────────────
function StockRow({ stock, onSelect }) {
    return /*#__PURE__*/ React.createElement(
        "div",
        {
            className: "stock-row",
            style: { cursor: "pointer" },
            onClick: () => onSelect && onSelect(stock.ticker),
        },
    /*#__PURE__*/ React.createElement(
            "div",
            null,
      /*#__PURE__*/ React.createElement(
                "div",
                { className: "stock-ticker" },
                stock.ticker?.replace(".NS", "")
            ),
      /*#__PURE__*/ React.createElement("div", { className: "stock-name" }, stock.name)
        ),
    /*#__PURE__*/ React.createElement("div", { style: { flex: 1 } }),
    /*#__PURE__*/ React.createElement(
            "div",
            null,
      /*#__PURE__*/ React.createElement(
                "div",
                { className: "stock-price" },
                "₹",
                stock.price?.toLocaleString("en-IN")
            ),
      /*#__PURE__*/ React.createElement(
                "div",
                { style: { fontSize: 11, color: "var(--text-muted)", textAlign: "right" } },
                stock.change >= 0 ? "+" : "",
                stock.change?.toFixed(2)
            )
        ),
/*#__