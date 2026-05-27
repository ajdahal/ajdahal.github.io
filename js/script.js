(() => {
  const root = document.documentElement;
  const toggle = document.querySelector(".theme-toggle");
  const label = toggle?.querySelector(".theme-toggle-text");
  const icon = toggle?.querySelector(".theme-toggle-icon");

  function getStoredTheme() {
    try {
      return localStorage.getItem("theme");
    } catch {
      return null;
    }
  }

  function storeTheme(theme) {
    try {
      localStorage.setItem("theme", theme);
    } catch {
      /* Theme still works for the current page if storage is unavailable. */
    }
  }

  function getPreferredTheme() {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function setTheme(theme) {
    const isDark = theme === "dark";
    root.setAttribute("data-theme", theme);
    toggle?.setAttribute("aria-pressed", String(isDark));
    toggle?.setAttribute("aria-label", isDark ? "Switch to light theme" : "Switch to dark theme");

    if (label) label.textContent = isDark ? "Light" : "Dark";
    if (icon) icon.textContent = isDark ? "L" : "D";
  }

  setTheme(getStoredTheme() || root.getAttribute("data-theme") || getPreferredTheme());

  toggle?.addEventListener("click", () => {
    const nextTheme = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    storeTheme(nextTheme);
  });
})();
