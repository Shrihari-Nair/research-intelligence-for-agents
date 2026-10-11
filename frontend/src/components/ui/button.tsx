import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 rounded-xl text-sm font-semibold transition-all duration-200 active:scale-[0.97] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-volt focus-visible:ring-offset-2 focus-visible:ring-offset-ink disabled:pointer-events-none disabled:opacity-40",
  {
    variants: {
      variant: {
        default: "bg-volt text-ink hover:bg-volt-soft hover:shadow-glow hover:-translate-y-px",
        outline: "border border-white/15 bg-white/[0.03] text-fg hover:border-volt/50 hover:bg-volt/10 hover:text-volt",
        destructive: "bg-rose-500 text-white hover:bg-rose-400 hover:shadow-[0_8px_30px_-8px_rgba(244,63,94,.6)]",
        ghost: "text-fg-dim hover:bg-white/5 hover:text-fg",
      },
      size: { default: "h-10 px-5", sm: "h-8 px-3.5 text-xs" },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, ...props }, ref) => (
  <button ref={ref} className={cn(buttonVariants({ variant, size }), className)} {...props} />
));
Button.displayName = "Button";
