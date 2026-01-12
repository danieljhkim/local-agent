/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'tier-0': '#10b981', // green for read-only
        'tier-1': '#f59e0b', // yellow for drafting
        'tier-2': '#ef4444', // red for side-effects
      },
    },
  },
  plugins: [],
}
