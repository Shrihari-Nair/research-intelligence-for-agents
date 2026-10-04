import "./globals.css";
import Link from "next/link";
import type { ReactNode } from "react";

export const metadata = { title: "Research Intelligence OS" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900">
        <header className="border-b bg-white">
          <nav className="mx-auto flex max-w-7xl items-center gap-6 px-6 py-3 text-sm">
            <Link href="/" className="font-semibold">Research Intelligence OS</Link>
            <Link href="/" className="text-slate-600 hover:text-slate-900">New research</Link>
            <Link href="/history" className="text-slate-600 hover:text-slate-900" data-testid="nav-history">History</Link>
          </nav>
        </header>
        <div className="mx-auto max-w-7xl px-6 py-6">{children}</div>
      </body>
    </html>
  );
}
