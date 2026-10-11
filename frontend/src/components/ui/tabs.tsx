"use client";
import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import { cn } from "@/lib/utils";

export const Tabs = TabsPrimitive.Root;
export function TabsList({ className, ...props }: React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>) {
  return <TabsPrimitive.List className={cn("inline-flex items-center gap-1 rounded-2xl border border-line bg-ink-2/80 p-1 backdrop-blur", className)} {...props} />;
}
export function TabsTrigger({ className, ...props }: React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>) {
  return (
    <TabsPrimitive.Trigger
      className={cn("rounded-xl px-4 py-1.5 text-sm font-medium text-fg-dim transition-all duration-200 hover:text-fg data-[state=active]:bg-volt data-[state=active]:text-ink data-[state=active]:shadow-glow", className)}
      {...props}
    />
  );
}
export const TabsContent = TabsPrimitive.Content;
