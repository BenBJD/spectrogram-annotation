import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

# Absolute import from your package
from spec_annotate.main_window import MainWindow


def resource_path(relative_path: str) -> str:
    """
    Ensures assets (like icons) are located correctly in both development
    (relative path) and PyInstaller-bundled environments (using sys._MEIPASS).
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Fallback for development environment
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def run_app() -> int:
    """
    Initialises and runs the PySide6 application.
    """
    app = QApplication(sys.argv)

    # --- ICON SETUP ---
    icon_file = resource_path("assets/spectrogram.svg")

    app_icon = QIcon(icon_file)
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    else:
        # Prints a warning if the icon is not found, but allows the app to proceed
        print(
            f"Warning: Icon file not found or failed to load: {icon_file}. Using default.")

    win = MainWindow()

    # Ensure the main window uses the application icon
    win.setWindowIcon(app.windowIcon())
    win.show()

    # Execute the application loop
    return app.exec()


if __name__ == "__main__":
    # The entry point should only call the main execution function
    sys.exit(run_app())