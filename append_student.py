# --------------------------
# ИМПОРТЫ
# --------------------------
import sys
import csv
import json
from typing import List, Tuple, Optional
import os
from pathlib import Path
from PyQt5.QtCore import QSettings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# seaborn — опционально
try:
    import seaborn as sns
except ImportError:
    sns = None

# Статистика и метрики
from scipy.stats import t as tdist, pearsonr
from sklearn.metrics import r2_score, mean_squared_error

# === Pillow + HEIC ===
from PIL import Image, UnidentifiedImageError
try:
    # Новые версии pillow-heif (>=0.13) используют register_heif_opener
    from pillow_heif import register_heif_opener as _register_heif
except ImportError:
    try:
        # Старые версии — register_heif
        from pillow_heif import register_heif as _register_heif
    except ImportError:
        _register_heif = None

if _register_heif is not None:
    try:
        _register_heif()
    except Exception as e:
        print(f"[WARNING] Не удалось зарегистрировать HEIF/HEIC декодер: {e}")
else:
    print("[INFO] pillow-heif не установлен, HEIC может не открыться.")

# === PyQt5 ===
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QTabWidget, QListWidget, QListWidgetItem, QGroupBox, QFormLayout, QSpinBox,
    QDoubleSpinBox, QCheckBox, QTextEdit,  QCheckBox, QTextEdit, QInputDialog
)




# --------------------------
# ОКНО: Вставка изображения 
# --------------------------
class ImagePane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pixmap = None
        self.current_path = None

        main = QVBoxLayout(self)

        # Панель кнопок
        btns = QHBoxLayout()
        self.open_btn = QPushButton("Открыть фото…")
        self.save_btn = QPushButton("Сохранить как PNG…")
        self.fit_box = QCheckBox("Вписать в окно")
        self.fit_box.setChecked(True)
        btns.addWidget(self.open_btn)
        btns.addWidget(self.save_btn)
        btns.addStretch()
        btns.addWidget(self.fit_box)
        main.addLayout(btns)

        # Поле предпросмотра
        self.preview = QLabel("Нет изображения")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("QLabel { background:#111; color:#888; border:1px solid #333; }")
        self.preview.setMinimumWidth(360)
        main.addWidget(self.preview, 1)

        # Сигналы
        self.open_btn.clicked.connect(self.open_image_dialog)
        self.save_btn.clicked.connect(self.save_as_png)
        self.fit_box.stateChanged.connect(self._refresh_view)

        # Разрешим перетаскивание файла (drag&drop)
        self.setAcceptDrops(True)

    # === Drag & Drop ===
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.load_image(path)
                break

    # === Диалог открытия ===
    def open_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть изображение",
            "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.webp *.heic *.HEIC)"
        )
        if path:
            self.load_image(path)

    # === Загрузка с поддержкой HEIC (план Б при неудаче) ===
   # === Загрузка с поддержкой HEIC (план Б при неудаче) ===
    def load_image(self, path: str):
        try:
            img = Image.open(path)
            img.load()
        except UnidentifiedImageError:
            try:
                from pillow_heif import open_heif, read_heif
            except Exception as e:
                QMessageBox.critical(self, "Ошибка загрузки",
                                     f"HEIC не поддерживается (нет pillow-heif):\n{e}")
                return
            try:
                try:
                    img = open_heif(path).to_pillow()
                except Exception:
                    heif_file = read_heif(path)
                    img = Image.frombytes(
                        heif_file.mode, heif_file.size, heif_file.data,
                        "raw", heif_file.mode, heif_file.stride
                    )
            except Exception as e:
                QMessageBox.critical(self, "Ошибка загрузки",
                                     f"Не удалось декодировать HEIC:\n{path}\n\n{e}")
                return
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки",
                                 f"Не удалось открыть файл:\n{path}\n\n{e}")
            return

        # Нормализуем формат и показываем
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")

        qimg = self._pil_to_qimage(img)   # <-- используем свой конвертер
        self.current_pixmap = QtGui.QPixmap.fromImage(qimg)
        self.current_path = path
        self._refresh_view()

    # === Конвертация PIL.Image -> QImage без ImageQt ===
    def _pil_to_qimage(self, pil_img: "Image.Image") -> QtGui.QImage:
        arr = np.array(pil_img.convert("RGBA"), copy=False)
        h, w, _ = arr.shape
        qimg = QtGui.QImage(arr.data, w, h, 4 * w, QtGui.QImage.Format_RGBA8888)
        return qimg.copy()


    # === Сохранение как PNG ===
    def save_as_png(self):
        if not self.current_pixmap:
            QMessageBox.information(self, "Нет изображения", "Сначала откройте файл.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить как PNG", "", "PNG (*.png)")
        if not path:
            return
        ok = self.current_pixmap.save(path, "PNG")
        if ok:
            QMessageBox.information(self, "Готово", "Сохранено.")
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось сохранить файл.")

    # === Обновление отображения (вписывание/масштаб) ===
    def _refresh_view(self):
        if not self.current_pixmap:
            self.preview.setText("Нет изображения")
            self.preview.setPixmap(QtGui.QPixmap())
            return
        if self.fit_box.isChecked():
            area = self.preview.size()
            scaled = self.current_pixmap.scaled(
                area, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.preview.setPixmap(scaled)
        else:
            self.preview.setPixmap(self.current_pixmap)

    # Автообновление при ресайзе
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh_view()



# --------------------------
# Вкладка 1: ДАННЫЕ (все значения числовые; CSV/XLSX загрузка)
# --------------------------
class DataTab(QWidget):
    HEADERS = [
        "Ширина (px)", "Высота (px)", "Ширина (%)", "Высота (%)", "Аспект (Ш/В)",
        "Дистанция (м)", "Яркость",
        "Время суток", "Погода", "Качество фото",
        "Абс дист"
    ]

    NUMERIC_COLUMNS = {
        "Ширина (px)", "Высота (px)", "Ширина (%)", "Высота (%)", "Аспект (Ш/В)",
        "Дистанция (м)", "Яркость",
        "Время суток", "Погода", "Качество фото",
        "Абс дист"
    }

    # Коды категорий и порядок
    TIME_OF_DAY = [("Утро", 1), ("День", 2), ("Вечер", 3), ("Ночь", 4)]
    WEATHER = [("Ясно", 1), ("Облачно", 2), ("Дождь", 3), ("Снег", 4), ("Туман", 5)]
    PHOTO_QUALITY = [("Хорошее", 1), ("Среднее", 2), ("Плохое", 3)]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.csv_delimiter = ";"   # можно менять во вкладке «Настройки»
        self.decimals = 3

        main_layout = QVBoxLayout(self)
        form_layout = QHBoxLayout()
        col1 = QVBoxLayout()
        col2 = QVBoxLayout()

        # Поля ввода (только числа)
        self.width_px = QLineEdit();   self.width_px.setPlaceholderText("Только число")
        self.height_px = QLineEdit();  self.height_px.setPlaceholderText("Только число")
        self.width_perc = QLineEdit(); self.width_perc.setPlaceholderText("Только число")
        self.height_perc = QLineEdit();self.height_perc.setPlaceholderText("Только число")
        self.aspect = QLineEdit();     self.aspect.setPlaceholderText("Только число")
        self.distance = QLineEdit();   self.distance.setPlaceholderText("Только число")
        self.brightness = QLineEdit(); self.brightness.setPlaceholderText("Только число")
        self.abs_dist = QLineEdit();   self.abs_dist.setPlaceholderText("Только число")

        validator = QtGui.QDoubleValidator(0.0, 1e12, 6)
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        for f in [self.width_px, self.height_px, self.width_perc, self.height_perc,
                  self.aspect, self.distance, self.brightness, self.abs_dist]:
            f.setValidator(validator)

        # Комбо с кодами (в таблицу пишем числа)
        self.time_of_day = QComboBox()
        for name, code in self.TIME_OF_DAY:
            self.time_of_day.addItem(f"{name} ({code})", code)
        self.weather = QComboBox()
        for name, code in self.WEATHER:
            self.weather.addItem(f"{name} ({code})", code)
        self.photo_quality = QComboBox()
        for name, code in self.PHOTO_QUALITY:
            self.photo_quality.addItem(f"{name} ({code})", code)

        # Разметка форм
        col1.addWidget(QLabel("Ширина (px)"));  col1.addWidget(self.width_px)
        col1.addWidget(QLabel("Высота (px)"));  col1.addWidget(self.height_px)
        col1.addWidget(QLabel("Ширина (%)"));   col1.addWidget(self.width_perc)
        col1.addWidget(QLabel("Высота (%)"));   col1.addWidget(self.height_perc)
        col1.addWidget(QLabel("Аспект (Ш/В)")); col1.addWidget(self.aspect)
        col1.addWidget(QLabel("Дистанция (м)"));col1.addWidget(self.distance)

        col2.addWidget(QLabel("Яркость"));      col2.addWidget(self.brightness)
        col2.addWidget(QLabel("Время суток"));  col2.addWidget(self.time_of_day)
        col2.addWidget(QLabel("Погода"));       col2.addWidget(self.weather)
        col2.addWidget(QLabel("Качество фото"));col2.addWidget(self.photo_quality)
        col2.addWidget(QLabel("Абс дист"));     col2.addWidget(self.abs_dist)

        form_layout.addLayout(col1); form_layout.addLayout(col2)
        main_layout.addLayout(form_layout)

        # Кнопки
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Добавить")
        self.clear_btn = QPushButton("Сбросить")
        self.save_btn = QPushButton("Сохранить в CSV")
        self.load_btn = QPushButton("Загрузить CSV/XLSX")
        self.delete_btn = QPushButton("Удалить выбранное")
        for b in [self.ok_btn, self.clear_btn, self.save_btn, self.load_btn, self.delete_btn]:
            button_layout.addWidget(b)
        main_layout.addLayout(button_layout)

        # Таблица
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.DoubleClicked)
        main_layout.addWidget(self.table)

        # Сигналы
        self.ok_btn.clicked.connect(self.add_row)
        self.clear_btn.clicked.connect(self.clear_fields)
        self.save_btn.clicked.connect(self.save_csv)
        self.load_btn.clicked.connect(self.load_csv)
        self.delete_btn.clicked.connect(self.delete_row)

    # ---------- API ----------
    def get_headers(self) -> List[str]:
        return [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]

    def _current_code(self, combo: QComboBox) -> int:
        return int(combo.currentData())

    def get_numpy_data(self, selected_features: List[str], target: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        headers = self.get_headers()
        col_index = {h: i for i, h in enumerate(headers)}

        rows = []
        for r in range(self.table.rowCount()):
            tgt_item = self.table.item(r, col_index[target])
            try:
                _ = float(tgt_item.text().strip()) if tgt_item else None
            except Exception:
                continue

            row_vals = []
            for h in headers:
                item = self.table.item(r, col_index[h])
                row_vals.append(item.text().strip() if item else "")
            rows.append(row_vals)

        if not rows:
            raise ValueError("Нет валидных строк в таблице для обучения.")

        X_list, y_list = [], []
        for row in rows:
            x_vec = []
            for feat in selected_features:
                idx = col_index[feat]
                try:
                    x_vec.append(float(row[idx]) if row[idx] != "" else np.nan)
                except Exception:
                    x_vec.append(np.nan)
            y_val = float(row[col_index[target]])
            X_list.append(x_vec)
            y_list.append(y_val)

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=float)

        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])

        return X, y, list(selected_features)

    # ---------- Действия ----------
    def add_row(self):
        # требуем все числовые поля
        fields = [
            self.width_px, self.height_px, self.width_perc, self.height_perc,
            self.aspect, self.distance, self.brightness, self.abs_dist
        ]
        for f in fields:
            if not f.text().strip():
                QMessageBox.warning(self, "Ошибка", "Заполните все числовые поля!")
                return

        # коды категорий → числа
        tod_code = self._current_code(self.time_of_day)
        weather_code = self._current_code(self.weather)
        quality_code = self._current_code(self.photo_quality)

        row = self.table.rowCount()
        self.table.insertRow(row)
        values = [
            self.width_px.text(), self.height_px.text(),
            self.width_perc.text(), self.height_perc.text(),
            self.aspect.text(), self.distance.text(), self.brightness.text(),
            str(tod_code), str(weather_code), str(quality_code),
            self.abs_dist.text()
        ]
        for col, value in enumerate(values):
            self.table.setItem(row, col, QTableWidgetItem(value))
        self.clear_fields()

    def clear_fields(self):
        for f in [self.width_px, self.height_px, self.width_perc, self.height_perc,
                  self.aspect, self.distance, self.brightness, self.abs_dist]:
            f.clear()
        self.time_of_day.setCurrentIndex(0)   # Утро (1)
        self.weather.setCurrentIndex(0)       # Ясно (1)
        self.photo_quality.setCurrentIndex(0) # Хорошее (1)

    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить CSV", "", "CSV files (*.csv)")
        if not path:
            return
        with open(path, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=self.csv_delimiter)
            writer.writerow([self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())])
            for row in range(self.table.rowCount()):
                writer.writerow([
                    self.table.item(row, col).text() if self.table.item(row, col) else ""
                    for col in range(self.table.columnCount())
                ])
        QMessageBox.information(self, "Успех", "Таблица сохранена в CSV!")

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть файл данных", "",
            "CSV/Excel files (*.csv *.xlsx)"
        )
        if not path:
            return

        # читаем файл
        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path, delimiter=self.csv_delimiter, encoding="utf-8")
            elif path.lower().endswith(".xlsx"):
                try:
                    import openpyxl  # noqa: F401
                except ImportError:
                    QMessageBox.critical(self, "Ошибка чтения файла",
                                         "Нужен пакет 'openpyxl' (pip install openpyxl).")
                    return
                with pd.ExcelFile(path) as xls:
                    sheets = xls.sheet_names
                    if "Данные" in sheets:
                        sheet = "Данные"
                    elif len(sheets) == 1:
                        sheet = sheets[0]
                    else:
                        from PyQt5.QtWidgets import QInputDialog
                        sheet, ok = QInputDialog.getItem(self, "Выбор листа Excel",
                                                         "Выберите лист:", sheets, 0, False)
                        if not ok:
                            return
                    df = pd.read_excel(xls, sheet_name=sheet)
            else:
                QMessageBox.warning(self, "Ошибка", "Неподдерживаемый формат файла.")
                return
        except Exception as e:
            QMessageBox.critical(self, "Ошибка чтения файла", f"Не удалось прочитать файл:\n{e}")
            return

        if df.empty:
            QMessageBox.warning(self, "Ошибка", "Файл пуст.")
            return

        # выравниваем столбцы к порядку HEADERS (лишние игнорируем, недостающие создаём)
        for h in self.HEADERS:
            if h not in df.columns:
                df[h] = pd.NA
        df = df[self.HEADERS]

        # числовые столбцы: приводим запятую к точке и парсим в float
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = (df[col].astype(str)
                                   .str.replace(",", ".", regex=False))
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # в таблицу — значения
        self.table.setColumnCount(len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setRowCount(0)
        for _, row_s in df.iterrows():
            row = self.table.rowCount()
            self.table.insertRow(row)
            for col, value in enumerate(row_s.tolist()):
                text = "" if (pd.isna(value)) else str(value)
                self.table.setItem(row, col, QTableWidgetItem(text))

        QMessageBox.information(self, "Успех", "Данные успешно загружены!")

    def delete_row(self):
        """Удаляет выделенные строки из таблицы."""
        selected = self.table.selectionModel().selectedRows()
        for idx in sorted(selected, key=lambda i: i.row(), reverse=True):
            self.table.removeRow(idx.row())

# --------------------------
# Вкладка 2: РЕГРЕССИОННАЯ МОДЕЛЬ (без QStandardPaths)
# --------------------------
class RegressionTab(QWidget):
    """
    Пайплайн:
      1) t-критерий значимости (парная корреляция с Y)
      2) Итеративное удаление мультиколлинеарности по |corr| > threshold
      3) OLS (закрытая форма) + уравнение
      4) Метрики: Multiple R, R^2, Adj R^2, RMSE, N
      5) Графики: t-значения, корреляционная матрица, pairplot
      + сохранение графика в PNG
      + персистентность α и threshold через INI (QSettings)
    """
    SETTINGS_SECTION = "regression"
    KEY_ALPHA = "alpha"
    KEY_MCTHRESH = "mc_threshold"

    def __init__(self, data_tab: DataTab, parent=None):
        super().__init__(parent)
        self.data_tab = data_tab

        # --- QSettings в INI-файл (без QCoreApplication и без QStandardPaths) ---
        ini_path = self._compute_settings_path()
        self.settings = QSettings(ini_path, QSettings.IniFormat)

        # ---- UI ----
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        params = QGroupBox("Параметры анализа")
        form = QFormLayout()

        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.001, 0.50)
        self.alpha.setSingleStep(0.01)
        self.alpha.setDecimals(3)

        self.mc_threshold = QDoubleSpinBox()
        self.mc_threshold.setRange(0.10, 0.999)
        self.mc_threshold.setDecimals(3)
        self.mc_threshold.setSingleStep(0.05)

        # Загрузка сохранённых значений (или дефолтов)
        self._load_settings_defaults(alpha_default=0.10, mc_default=0.70)
        # Автосохранение при изменении
        self.alpha.valueChanged.connect(self._persist_settings)
        self.mc_threshold.valueChanged.connect(self._persist_settings)

        form.addRow("Уровень значимости α:", self.alpha)
        form.addRow("Порог |corr| для мультиколл.:", self.mc_threshold)
        params.setLayout(form)

        # Кнопки
        self.run_btn = QPushButton("Анализ (t → MC → OLS)")
        self.plot_t_btn = QPushButton("График t-значений")
        self.plot_corr_btn = QPushButton("Корреляционная матрица")
        self.plot_pair_btn = QPushButton("Pairplot (после MC)")
        self.save_plot_btn = QPushButton("Сохранить график .png")
        self.save_plot_btn.setEnabled(False)
        self.save_report_btn = QPushButton("Сохранить отчёт .txt")

        left.addWidget(params)
        left.addWidget(self.run_btn)
        left.addWidget(self.plot_t_btn)
        left.addWidget(self.plot_corr_btn)
        left.addWidget(self.plot_pair_btn)
        left.addWidget(self.save_plot_btn)
        left.addStretch()
        left.addWidget(self.save_report_btn)

        right = QVBoxLayout()
        self.out_box = QGroupBox("Результаты")
        v = QVBoxLayout()
        self.out_text = QTextEdit()
        self.out_text.setReadOnly(True)
        v.addWidget(self.out_text)
        self.out_box.setLayout(v)
        right.addWidget(self.out_box)

        root.addLayout(left, 2)
        root.addLayout(right, 3)

        # Сигналы
        self.run_btn.clicked.connect(self.run_analysis)
        self.plot_t_btn.clicked.connect(self.plot_t_values)
        self.plot_corr_btn.clicked.connect(self.plot_corr)
        self.plot_pair_btn.clicked.connect(self.plot_pairplot)
        self.save_plot_btn.clicked.connect(self.save_last_plot)
        self.save_report_btn.clicked.connect(self.save_report)

        # Состояние
        self.last_df: Optional[pd.DataFrame] = None
        self.last_target: Optional[str] = None
        self.last_all_num_features: list[str] = []
        self.last_significant: list[str] = []
        self.last_pruned: list[str] = []
        self.last_t_values: list[float] = []
        self.last_r_values: list[float] = []
        self.last_feature_names: list[str] = []
        self.last_t_critical: Optional[float] = None
        self.last_equation: Optional[str] = None
        self.last_report: Optional[str] = None
        self.last_y_pred: Optional[np.ndarray] = None
        self.last_fig = None  # matplotlib.figure.Figure

    # --------- INI-файл для настроек ---------
    @staticmethod
    def _compute_settings_path() -> str:
        """
        Возвращает полный путь к INI с настройками:
        - Windows: %APPDATA%/AnalyticsApp/analytics_app.ini
        - Linux/macOS: $XDG_CONFIG_HOME/AnalyticsApp/... или ~/.config/AnalyticsApp/...
        """
        try:
            if os.name == "nt":
                base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
            else:
                base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
            app_dir = Path(base) / "AnalyticsApp"
            app_dir.mkdir(parents=True, exist_ok=True)
            return str(app_dir / "analytics_app.ini")
        except Exception:
            # Фолбэк — домашняя папка
            return str(Path.home() / ".analytics_app.ini")

    # --------- Настройки (QSettings) ---------
    def _load_settings_defaults(self, alpha_default: float, mc_default: float):
        try:
            a = float(self.settings.value(f"{self.SETTINGS_SECTION}/{self.KEY_ALPHA}", alpha_default))
        except Exception:
            a = alpha_default
        try:
            m = float(self.settings.value(f"{self.SETTINGS_SECTION}/{self.KEY_MCTHRESH}", mc_default))
        except Exception:
            m = mc_default
        self.alpha.setValue(a)
        self.mc_threshold.setValue(m)

    def _persist_settings(self):
        self.settings.setValue(f"{self.SETTINGS_SECTION}/{self.KEY_ALPHA}", self.alpha.value())
        self.settings.setValue(f"{self.SETTINGS_SECTION}/{self.KEY_MCTHRESH}", self.mc_threshold.value())
        self.settings.sync()

    # --------- Утилиты ---------
    @staticmethod
    def _coerce_float(x: str) -> Optional[float]:
        try:
            return float(str(x).replace(",", "."))
        except Exception:
            return None

    def _table_to_dataframe(self) -> pd.DataFrame:
        headers = self.data_tab.get_headers()
        rows = []
        for r in range(self.data_tab.table.rowCount()):
            row = {}
            for c, h in enumerate(headers):
                item = self.data_tab.table.item(r, c)
                text = item.text().strip() if item else ""
                if h in self.data_tab.NUMERIC_COLUMNS:
                    row[h] = self._coerce_float(text)
                else:
                    row[h] = text
            rows.append(row)
        df = pd.DataFrame(rows)
        if "Ширина (px)" in df.columns and "Высота (px)" in df.columns:
            df["Площадь bbox (px²)"] = (
                pd.to_numeric(df["Ширина (px)"], errors="coerce")
                * pd.to_numeric(df["Высота (px)"], errors="coerce")
            )
        return df

    def _initial_features(self, df: pd.DataFrame) -> list[str]:
        wanted = [
            "Ширина (px)", "Ширина (%)", "Высота (px)", "Высота (%)",
            "Аспект (Ш/В)", "Дистанция (м)", "Яркость",
            "Погода", "Время суток", "Качество фото"
        ]
        return [c for c in wanted if c in df.columns]

    def _target_name(self, df: pd.DataFrame) -> str:
        if "Абс дист" in df.columns:
            return "Абс дист"
        raise ValueError("В таблице нет колонки 'Абс дист'.")

    @staticmethod
    def _safe_pearsonr(x, y) -> float:
        try:
            r, _ = pearsonr(x, y)
            return 0.0 if np.isnan(r) else float(r)
        except Exception:
            return 0.0

    def _drop_multicollinear_by_threshold(
        self,
        df: pd.DataFrame,
        features: list[str],
        y: pd.Series,
        threshold: float,
        verbose_lines: list[str],
    ) -> list[str]:
        feats = [f for f in features if f in df.columns]
        feats = [f for f in feats if np.issubdtype(df[f].dtype, np.number)]
        feats = list(dict.fromkeys(feats))
        if len(feats) <= 1:
            return feats

        y_vals = y.values if hasattr(y, "values") else np.asarray(y)

        while True:
            if len(feats) <= 1:
                break
            Xcorr = df[feats].corr().abs()
            np.fill_diagonal(Xcorr.values, 0.0)
            max_val = Xcorr.values.max()
            if not np.isfinite(max_val) or max_val <= threshold:
                break
            idx = np.argwhere(Xcorr.values == max_val)[0]
            i, j = idx[0], idx[1]
            f1, f2 = feats[i], feats[j]
            r1 = abs(self._safe_pearsonr(df[f1].values, y_vals))
            r2 = abs(self._safe_pearsonr(df[f2].values, y_vals))
            if r1 < r2:
                drop = f1
            elif r2 < r1:
                drop = f2
            else:
                mean1 = Xcorr.loc[f1, [c for c in feats if c != f1]].mean()
                mean2 = Xcorr.loc[f2, [c for c in feats if c != f2]].mean()
                drop = f1 if mean1 >= mean2 else f2
            verbose_lines.append(
                f"[MC] |corr({f1},{f2})|={max_val:.3f} > {threshold}. "
                f"|corr({f1},Y)|={r1:.3f}, |corr({f2},Y)|={r2:.3f} → удаляем: {drop}"
            )
            feats.remove(drop)
        return feats

    @staticmethod
    def _ols_closed_form(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        try:
            theta_hat = np.linalg.inv(X_b.T @ X_b) @ (X_b.T @ y)
        except np.linalg.LinAlgError:
            theta_hat = np.linalg.pinv(X_b.T @ X_b) @ (X_b.T @ y)
        intercept = float(theta_hat[0])
        coef = theta_hat[1:].astype(float)
        y_pred = X_b @ theta_hat
        return coef, intercept, y_pred

    @staticmethod
    def _metrics_excel_like(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> tuple[float, float, float, int, float]:
        r2 = r2_score(y_true, y_pred)
        multiple_r = np.sqrt(max(0.0, r2))
        n = len(y_true)
        adj_r2 = 1 - (1 - r2) * (n - 1) / max(1, (n - k - 1))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return multiple_r, r2, adj_r2, n, rmse

    # --------- Основной анализ ---------
    def run_analysis(self):
        try:
            df = self._table_to_dataframe()
            target = self._target_name(df)
            if df.empty:
                raise ValueError("Нет данных в таблице.")
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()

            alpha = float(self.alpha.value())
            threshold = float(self.mc_threshold.value())

            initial_features = self._initial_features(df)
            num_features = [f for f in initial_features if np.issubdtype(df[f].dtype, np.number)]
            if not num_features:
                raise ValueError("Нет числовых признаков для анализа.")

            y = pd.to_numeric(df[target], errors="coerce")
            work = df[num_features + [target]].dropna(subset=[target]).copy()
            n = len(work)
            if n < 3:
                raise ValueError("Слишком мало валидных наблюдений (нужно ≥ 3).")

            dfree = max(n - 2, 1)
            t_critical = abs(tdist.ppf(alpha / 2, dfree))

            # --- t-значения и r
            t_values, r_values, feat_names, significant = [], [], [], []
            y_vals = work[target].values.astype(float)
            eps = 1e-12

            for f in num_features:
                x_vals = work[f].values.astype(float)
                mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                if mask.sum() < 3:
                    continue
                r_val, _ = pearsonr(x_vals[mask], y_vals[mask])
                denom = max(eps, 1.0 - r_val ** 2)
                t_stat = abs(r_val) * np.sqrt(max(0, dfree) / denom)
                t_values.append(float(t_stat))
                r_values.append(float(r_val))
                feat_names.append(f)
                if t_stat > t_critical:
                    significant.append(f)

            if not feat_names:
                raise ValueError("Не удалось вычислить t-статистики: нет валидных признаков.")

            # --- Мультиколлинеарность
            verbose_lines = []
            pruned = self._drop_multicollinear_by_threshold(
                work, significant, y, threshold=threshold, verbose_lines=verbose_lines
            )
            if not pruned:
                raise RuntimeError("После фильтрации мультиколлинеарности не осталось признаков. "
                                   "Ослабьте порог или проверьте данные.")

            # --- OLS
            X = work[pruned].to_numpy(dtype=float)
            y_vec = y_vals
            coef, intercept, y_pred = self._ols_closed_form(X, y_vec)

            # --- Метрики
            multiple_r, r2, adj_r2, n_obs, rmse = self._metrics_excel_like(y_vec, y_pred, k=X.shape[1])

            # --- Уравнение
            equation = f"{target} = {intercept:.6f}"
            for feature, w in zip(pruned, coef):
                equation += f" + ({float(w):.6f} × {feature})"

            # --- Отчёт
            lines = []
            lines.append(f"Критическое значение t при α = {alpha:.2f}, df = {dfree} → t_табл = {t_critical:.3f}\n")
            lines.append("Значимые по t-критерию признаки:")
            lines.append(", ".join(significant) if significant else "(нет)")
            lines.append("")
            if verbose_lines:
                lines.append("Удаление мультиколлинеарности (|r| > {0:.3f}):".format(threshold))
                for s in verbose_lines:
                    lines.append("  " + s)
                lines.append("")
            lines.append("Признаки после удаления мультиколлинеарности:")
            lines.append(", ".join(pruned))
            lines.append("")
            lines.append("Регрессионная модель (OLS):")
            lines.append(equation)
            lines.append("")
            lines.append("Регрессионная статистика:")
            lines.append(f"{'Множественный R:':30s} {multiple_r:.8f}")
            lines.append(f"{'R-квадрат:':30s} {r2:.8f}")
            lines.append(f"{'Нормированный R-квадрат:':30s} {adj_r2:.8f}")
            lines.append(f"{'Стандартная ошибка (RMSE):':30s} {rmse:.8f}")
            lines.append(f"{'Наблюдения:':30s} {n_obs}")

            report = "\n".join(lines)
            self.out_text.setPlainText(report)

            # Состояние
            self.last_df = work
            self.last_target = target
            self.last_all_num_features = feat_names
            self.last_significant = significant
            self.last_pruned = pruned
            self.last_t_values = t_values
            self.last_r_values = r_values
            self.last_feature_names = feat_names
            self.last_t_critical = t_critical
            self.last_equation = equation
            self.last_report = report
            self.last_y_pred = y_pred

            # Сброс последнего графика
            self.last_fig = None
            self.save_plot_btn.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка анализа", str(e))

    # --------- Графики ---------
    def plot_t_values(self):
        if not self.last_t_values or not self.last_feature_names:
            QMessageBox.information(self, "Нет данных", "Сначала выполните анализ.")
            return
        t_critical = self.last_t_critical or 0.0
        colors = ["tab:green" if t > t_critical else "lightgray" for t in self.last_t_values]
        fig = plt.figure(figsize=(10, 6))
        bars = plt.barh(self.last_feature_names, self.last_t_values, edgecolor='black', color=colors)
        plt.axvline(t_critical, linestyle='--', color='red', label=f"t крит = {t_critical:.2f}")
        plt.xlabel("|t|-значение")
        plt.title("Значимость признаков по t-критерию")
        plt.legend()
        plt.grid(True, axis='x')
        for bar, t_val in zip(bars, self.last_t_values):
            w = bar.get_width()
            plt.text(w + 0.05, bar.get_y() + bar.get_height()/2, f"{t_val:.2f}", va='center', fontsize=9)
        plt.tight_layout()
        plt.show(block=False)
        self.last_fig = fig
        self.save_plot_btn.setEnabled(True)

    def plot_corr(self):
        if self.last_df is None or self.last_target is None:
            QMessageBox.information(self, "Нет данных", "Сначала выполните анализ.")
            return
        cols = self.last_pruned + [self.last_target] if self.last_pruned else self.last_all_num_features + [self.last_target]
        cols = [c for c in cols if c in self.last_df.columns]
        if len(cols) < 2:
            QMessageBox.information(self, "Нет данных", "Недостаточно колонок для корреляции.")
            return
        corr = self.last_df[cols].corr()
        fig = plt.figure(figsize=(10, 8))
        if sns is not None:
            ax = sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
            ax.set_title("Корреляционная матрица (после отбора)")
        else:
            plt.imshow(corr, cmap="coolwarm")
            plt.colorbar()
            plt.title("Корреляционная матрица (seaborn не найден)")
            plt.xticks(range(len(cols)), cols, rotation=45, ha='right')
            plt.yticks(range(len(cols)), cols)
        plt.tight_layout()
        plt.show(block=False)
        self.last_fig = fig
        self.save_plot_btn.setEnabled(True)

    def plot_pairplot(self):
        if sns is None:
            QMessageBox.information(self, "Нет seaborn", "Пакет seaborn не найден. Установите его для pairplot.")
            return
        if self.last_df is None or self.last_target is None:
            QMessageBox.information(self, "Нет данных", "Сначала выполните анализ.")
            return
        cols = self.last_pruned + [self.last_target] if self.last_pruned else self.last_all_num_features + [self.last_target]
        cols = [c for c in cols if c in self.last_df.columns]
        if len(cols) < 2:
            QMessageBox.information(self, "Нет данных", "Недостаточно колонок для pairplot.")
            return
        g = sns.pairplot(self.last_df[cols])
        g.fig.suptitle("Попарные зависимости (после удаления мультиколлинеарности)", y=1.02)
        plt.show(block=False)
        self.last_fig = g.fig
        self.save_plot_btn.setEnabled(True)

    # --------- Сохранение графика ---------
    def save_last_plot(self):
        if self.last_fig is None:
            QMessageBox.information(self, "Нет графика", "Сначала постройте график.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить график", "", "PNG (*.png)")
        if not path:
            return
        try:
            self.last_fig.savefig(path, dpi=200, bbox_inches="tight")
            QMessageBox.information(self, "Готово", "График сохранён.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", str(e))

    # --------- Сохранение отчёта ---------
    def save_report(self):
        if not self.last_report:
            QMessageBox.information(self, "Нет отчёта", "Сначала выполните анализ.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить отчёт", "", "Text (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.last_report)
        QMessageBox.information(self, "Готово", "Отчёт сохранён.")


# --------------------------
# Вкладка 3: НАСТРОЙКИ
# --------------------------
class SettingsTab(QWidget):
    def __init__(self, data_tab: DataTab, parent=None):
        super().__init__(parent)
        self.data_tab = data_tab

        layout = QVBoxLayout(self)

        csv_group = QGroupBox("CSV")
        csv_form = QFormLayout()
        self.delim_combo = QComboBox()
        self.delim_combo.addItems([";", ",", "\t", "|"])
        self.precision_spin = QSpinBox()
        self.precision_spin.setRange(0, 10)
        self.precision_spin.setValue(self.data_tab.decimals)
        csv_form.addRow("Разделитель:", self.delim_combo)
        csv_form.addRow("Точность (знаков после запятой):", self.precision_spin)
        csv_group.setLayout(csv_form)

        fmt_group = QGroupBox("Отображение")
        fmt_form = QFormLayout()
        self.strict_numeric = QCheckBox("Требовать валидные числа при добавлении строки")
        self.strict_numeric.setChecked(True)
        fmt_form.addRow(self.strict_numeric)
        fmt_group.setLayout(fmt_form)

        apply_btn = QPushButton("Применить")
        apply_btn.clicked.connect(self.apply_settings)

        layout.addWidget(csv_group)
        layout.addWidget(fmt_group)
        layout.addWidget(apply_btn)
        layout.addStretch()

    def apply_settings(self):
        delim_map = {
            ";": ";",
            ",": ",",
            "\t": "\t",
            "|": "|"
        }
        self.data_tab.csv_delimiter = delim_map[self.delim_combo.currentText()]
        self.data_tab.decimals = self.precision_spin.value()
        QMessageBox.information(self, "ОК", "Настройки применены.")


# --------------------------
# Главное окно с вкладками
# --------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Аналитика объектов: данные + регрессия")
        self.resize(1400, 700)

        # Вместо чисто вертикального — сделаем горизонтальный сплит:
        root = QHBoxLayout(self)

        # Левая колонка: вкладки
        left = QVBoxLayout()
        self.tabs = QTabWidget()
        left.addWidget(self.tabs)

        left_wrap = QWidget()
        left_wrap.setLayout(left)

        # Вкладки
        self.data_tab = DataTab(self)
        self.tabs.addTab(self.data_tab, "Данные")

        self.reg_tab = RegressionTab(self.data_tab, self)
        self.tabs.addTab(self.reg_tab, "Регрессионная модель")

        self.settings_tab = SettingsTab(self.data_tab, self)
        self.tabs.addTab(self.settings_tab, "Настройки")

        # Правая колонка: постоянный предпросмотр изображений
        self.image_pane = ImagePane(self)

        # Добавляем в корневой layout
        root.addWidget(left_wrap, 3)      # вес левой части
        root.addWidget(self.image_pane, 2)  # вес правой части



def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
