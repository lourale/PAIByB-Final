"""
main.py
-------
Punto de entrada de la aplicación.
Inicializa QApplication y el controlador principal.
"""

import sys
from PyQt6.QtWidgets import QApplication
from controllers.main_controller import MainController


def main() -> None:
    """Función principal: arranca la aplicación Qt y el controlador raíz."""
    app = QApplication(sys.argv)
    app.setApplicationName("Clasificador de Imágenes")

    # Fondo global oscuro para toda la aplicación
    app.setStyleSheet(
        """
        QWidget {
            background-color: #16161e;
            color: #cdd6f4;
        }
        QTabWidget::pane {
            border: 1px solid #313244;
            background-color: #16161e;
        }
        QTabBar::tab {
            background-color: #1e1e2e;
            color: #6c7086;
            padding: 6px 16px;
            border: 1px solid #313244;
            border-bottom: none;
        }
        QTabBar::tab:selected {
            background-color: #16161e;
            color: #cdd6f4;
        }
        QGroupBox {
            border: 1px solid #313244;
            border-radius: 4px;
            margin-top: 8px;
            color: #a6adc8;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
        }
        QPushButton {
            background-color: #313244;
            color: #cdd6f4;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 4px 10px;
        }
        QPushButton:hover {
            background-color: #45475a;
        }
        QPushButton:disabled {
            color: #45475a;
        }
        QDoubleSpinBox, QSpinBox {
            background-color: #1e1e2e;
            color: #cdd6f4;
            border: 1px solid #45475a;
            border-radius: 3px;
            padding: 2px 4px;
        }
        QLabel {
            background-color: transparent;
        }
        QTextEdit {
            background-color: #1e1e2e;
            color: #cdd6f4;
            border: 1px solid #313244;
        }
        QTableWidget {
            background-color: #1e1e2e;
            color: #cdd6f4;
            gridline-color: #313244;
            border: 1px solid #313244;
        }
        QHeaderView::section {
            background-color: #313244;
            color: #a6adc8;
            border: 1px solid #45475a;
            padding: 4px;
        }
        QScrollBar:vertical {
            background: #1e1e2e;
            width: 10px;
        }
        QScrollBar::handle:vertical {
            background: #45475a;
            border-radius: 5px;
        }
        QCheckBox {
            spacing: 8px;
            color: #cdd6f4;
            background-color: transparent;
        }
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
            border: 2px solid #45475a;
            border-radius: 3px;
            background-color: #1e1e2e;
        }
        QCheckBox::indicator:checked {
            background-color: #89b4fa;
            border-color: #89b4fa;
        }
        QCheckBox::indicator:unchecked:hover {
            border-color: #89b4fa;
        }
        QCheckBox::indicator:disabled {
            background-color: #313244;
            border-color: #313244;
        }
        QRadioButton {
            spacing: 8px;
            color: #cdd6f4;
            background-color: transparent;
        }
        QRadioButton::indicator {
            width: 15px;
            height: 15px;
            border: 2px solid #45475a;
            border-radius: 8px;
            background-color: #1e1e2e;
        }
        QRadioButton::indicator:checked {
            background-color: #89b4fa;
            border-color: #89b4fa;
        }
        QRadioButton::indicator:disabled {
            background-color: #313244;
            border-color: #313244;
        }
        QComboBox {
            background-color: #313244;
            color: #cdd6f4;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 4px 8px;
        }
        QComboBox:hover {
            border-color: #89b4fa;
        }
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        QComboBox QAbstractItemView {
            background-color: #1e1e2e;
            color: #cdd6f4;
            border: 1px solid #45475a;
            selection-background-color: #45475a;
        }
        """
    )

    # El controlador principal se encarga de crear la vista y los sub-controladores
    controller = MainController()
    controller.show_maximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
