import "./globals.css";
import { Outfit, JetBrains_Mono } from "next/font/google";
import type { ReactNode } from "react";
import { Nav } from "@/components/Nav";

const sans = Outfit({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });

export const metadata = {
  title: "Research Intelligence OS",
  description: "Budget-capped research agents where every claim is verified against a quoted source.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body className="min-h-dvh font-sans">
        <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[70] focus:rounded-lg focus:bg-volt focus:px-3 focus:py-2 focus:text-ink">Skip to content</a>
        <Nav />
        <div id="main" className="mx-auto max-w-7xl px-6 pb-24 pt-10">{children}</div>
      </body>
    </html>
  );
}
