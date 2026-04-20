import type { Config } from 'tailwindcss';
import { colors } from './src/theme/tokens';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: colors.bg,
        card: colors.card,
        cardHover: colors.cardHover,
        border: colors.border,
        borderHover: colors.borderHover,
        gold: colors.gold,
        gold2: colors.gold2,
        teal: colors.teal,
        red: colors.red,
        amber: colors.amber,
        green: colors.green,
        blue: colors.blue,
        purple: colors.purple,
        text: colors.text,
        muted: colors.muted,
        faint: colors.faint,
      },
      fontFamily: {
        display: ['Playfair Display', 'serif'],
        body: ['DM Sans', 'sans-serif'],
        mono: ['DM Mono', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(240,180,41,0.24), 0 20px 60px rgba(5,10,24,0.45)',
      },
      animation: {
        pulseSoft: 'pulseSoft 2s ease-in-out infinite',
        fadeRise: 'fadeRise 0.6s ease both',
      },
      keyframes: {
        pulseSoft: {
          '0%, 100%': { transform: 'scale(1)', opacity: '1' },
          '50%': { transform: 'scale(1.05)', opacity: '0.9' },
        },
        fadeRise: {
          '0%': { opacity: '0', transform: 'translateY(18px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
