import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { DEFAULT: "#070a09", 2: "#0c110f", 3: "#121916", 4: "#1a231f" },
        line: "rgba(214,255,230,0.09)",
        fg: { DEFAULT: "#e9efe9", dim: "#9aa79f", faint: "#617068" },
        volt: { DEFAULT: "#c6ff3d", soft: "#d9ff85", deep: "#8fd100" },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(198,255,61,.35), 0 8px 40px -8px rgba(198,255,61,.45)",
        panel: "0 1px 0 0 rgba(255,255,255,.05) inset, 0 20px 50px -24px rgba(0,0,0,.9)",
      },
      keyframes: {
        "fade-up": { from: { opacity: "0", transform: "translateY(14px)" }, to: { opacity: "1", transform: "translateY(0)" } },
        "spin-slow": { to: { transform: "rotate(360deg)" } },
        ticker: { to: { transform: "translateX(-50%)" } },
        ping2: { "0%": { transform: "scale(1)", opacity: ".7" }, "100%": { transform: "scale(2.6)", opacity: "0" } },
        shimmer: { to: { backgroundPosition: "-200% 0" } },
      },
      animation: {
        "fade-up": "fade-up .7s cubic-bezier(.2,.7,.2,1) both",
        "spin-slow": "spin-slow 6s linear infinite",
        ticker: "ticker 38s linear infinite",
        ping2: "ping2 1.6s cubic-bezier(0,0,.2,1) infinite",
        shimmer: "shimmer 2.4s linear infinite",
      },
    },
  },
  plugins: [],
};
export default config;
