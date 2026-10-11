"use client";
import * as React from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { cn } from "@/lib/utils";

export const Dialog = DialogPrimitive.Root;
export const DialogClose = DialogPrimitive.Close;
export const DialogTitle = DialogPrimitive.Title;
export const DialogDescription = DialogPrimitive.Description;

export function DialogContent({ className, children, ...props }: React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content>) {
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-ink/70 backdrop-blur-md" />
      <DialogPrimitive.Content
        className={cn("panel fixed left-1/2 top-1/2 z-50 max-h-[85vh] w-[92vw] max-w-xl -translate-x-1/2 -translate-y-1/2 animate-fade-up overflow-auto rounded-3xl p-6 shadow-glow", className)}
        {...props}
      >
        {children}
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  );
}
