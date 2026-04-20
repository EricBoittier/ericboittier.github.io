(() => {
  const STORAGE_KEY = "site-theme";
  const root = document.documentElement;
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)");
  let button = null;

  function readStoredTheme() {
    try {
      const theme = localStorage.getItem(STORAGE_KEY);
      if (theme === "light" || theme === "dark") {
        return theme;
      }
    } catch (error) {
      // Ignore storage errors (private mode, restricted policies).
    }
    return null;
  }

  function writeStoredTheme(theme) {
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch (error) {
      // Ignore storage errors and keep behavior non-blocking.
    }
  }

  function effectiveTheme() {
    if (root.dataset.theme === "light" || root.dataset.theme === "dark") {
      return root.dataset.theme;
    }
    return prefersDark.matches ? "dark" : "light";
  }

  function updateButtonLabel() {
    if (!button) {
      return;
    }
    const current = effectiveTheme();
    const next = current === "dark" ? "light" : "dark";
    const label = next === "light" ? "Light mode" : "Dark mode";
    button.textContent = label;
    button.setAttribute("aria-label", `Switch to ${next} mode`);
    button.title = `Switch to ${next} mode`;
  }

  function applyStoredTheme() {
    const storedTheme = readStoredTheme();
    if (storedTheme) {
      root.dataset.theme = storedTheme;
    }
    updateButtonLabel();
  }

  function addButton() {
    const nav = document.querySelector("header nav");
    if (!nav) {
      return;
    }

    button = document.createElement("button");
    button.type = "button";
    button.className = "theme-toggle";
    button.addEventListener("click", () => {
      const next = effectiveTheme() === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      writeStoredTheme(next);
      updateButtonLabel();
    });

    nav.append(button);
    updateButtonLabel();
  }

  document.addEventListener("DOMContentLoaded", () => {
    applyStoredTheme();
    addButton();
  });
})();
