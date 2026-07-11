/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      screens: {
        sm: "576px",
        md: "768px",
        lg: "992px",
        xl: "1200px",
      },
      colors: {
        primary1: "#e91e63",
        primary2: "#9c27b0",
        secondary: "[#ba68c8]/70",

      }
    },
  },
  plugins: [],
}