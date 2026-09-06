"""
1MTR58 - LAB2
Simulador HMI de gripper virtual controlado mediante EOG

Uso para el informe:
    python 03_GripperVirtual.py

El archivo hmi_demo.csv se carga automáticamente y permite verificar
la correspondencia clase -> acción del gripper.

Uso opcional con un modelo entrenado:
    python 03_GripperVirtual.py --data archivo_features.csv --model modelo_grupo.joblib

En modo modelo, el CSV debe contener las mismas características
utilizadas durante el entrenamiento.
"""

import argparse
import math
import tkinter as tk
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

META_COLUMNS = {
    "label",
    "subject_id",
    "start_idx",
    "end_idx",
    "start_time",
    "end_time",
}

GESTURE_MAP = {
    1: "SACCADE OUTWARD",
    2: "RETURN SACCADE",
    3: "BLINK",
}

ACTION_MAP = {
    1: "ABRIR GRIPPER",
    2: "CERRAR GRIPPER",
    3: "ROTAR GRIPPER",
}

ACTION_DURATION_MS = 1000
ANIMATION_STEPS = 25


# ============================================================
# HMI
# ============================================================

class GripperApp:

    def __init__(self, root, df, model=None, interval_ms=1500):

        self.root = root
        self.df = df.reset_index(drop=True)
        self.model = model
        self.interval_ms = interval_ms

        self.idx = 0

        # Estado del gripper
        self.opening = 80.0
        self.angle = 0.0

        # Estado del ojo
        self.eye_class = None

        self.running = False
        self.animating = False

        # Dirección seleccionada para BLINK
        self.rotation_direction = tk.StringVar(value="HORARIO")

        root.title("1MTR58 - HMI EOG - Gripper Virtual")
        root.geometry("800x720")
        root.resizable(False, False)

        # ----------------------------------------------------
        # TÍTULO
        # ----------------------------------------------------

        tk.Label(
            root,
            text="CONTROL DE GRIPPER MEDIANTE SEÑALES EOG",
            font=("Arial", 16, "bold"),
        ).pack(pady=(12, 4))

        mode_text = (
            "MODO MODELO: clasificación mediante modelo entrenado"
            if self.model is not None
            else "MODO DEMO: la etiqueta del CSV se utiliza como comando"
        )

        tk.Label(
            root,
            text=mode_text,
            font=("Arial", 10),
        ).pack()

        # ----------------------------------------------------
        # CANVAS
        # ----------------------------------------------------

        self.canvas = tk.Canvas(
            root,
            width=760,
            height=420,
            bg="white",
            highlightthickness=1,
        )
        self.canvas.pack(padx=12, pady=12)

        # ----------------------------------------------------
        # INFORMACIÓN DE CLASIFICACIÓN
        # ----------------------------------------------------

        self.gesture_label = tk.Label(
            root,
            text="GESTO EOG: --",
            font=("Arial", 14, "bold"),
        )
        self.gesture_label.pack()

        self.prediction_label = tk.Label(
            root,
            text="Predicción: --",
            font=("Arial", 11),
        )
        self.prediction_label.pack()

        self.truth_label = tk.Label(
            root,
            text="Etiqueta real: --",
            font=("Arial", 11),
        )
        self.truth_label.pack()

        self.result_label = tk.Label(
            root,
            text="",
            font=("Arial", 11, "bold"),
        )
        self.result_label.pack(pady=(2, 8))

        # ----------------------------------------------------
        # SENTIDO DE ROTACIÓN
        # ----------------------------------------------------

        rotation_frame = tk.LabelFrame(
            root,
            text="Sentido de rotación para BLINK",
            padx=10,
            pady=6,
        )
        rotation_frame.pack(pady=5)

        tk.Radiobutton(
            rotation_frame,
            text="↻ Horario",
            variable=self.rotation_direction,
            value="HORARIO",
        ).pack(side="left", padx=15)

        tk.Radiobutton(
            rotation_frame,
            text="↺ Antihorario",
            variable=self.rotation_direction,
            value="ANTIHORARIO",
        ).pack(side="left", padx=15)

        # ----------------------------------------------------
        # CONTROLES
        # ----------------------------------------------------

        controls = tk.Frame(root)
        controls.pack(pady=(6, 6))

        tk.Button(
            controls,
            text="Iniciar",
            command=self.start,
            width=12,
        ).pack(side="left", padx=5)

        tk.Button(
            controls,
            text="Paso",
            command=self.step,
            width=12,
        ).pack(side="left", padx=5)

        tk.Button(
            controls,
            text="Reiniciar",
            command=self.reset,
            width=12,
        ).pack(side="left", padx=5)

        self.draw_scene()


    # ========================================================
    # FEATURES / MODELO
    # ========================================================

    def feature_columns(self):

        if self.model is not None and hasattr(
            self.model,
            "feature_names_in_",
        ):
            return list(self.model.feature_names_in_)

        return [
            c
            for c in self.df.columns
            if (
                c not in META_COLUMNS
                and pd.api.types.is_numeric_dtype(self.df[c])
            )
        ]


    def predict_row(self, row):

        # Modo demo
        if self.model is None:

            if "label" not in row.index:
                raise ValueError(
                    "El modo demo requiere una columna 'label'."
                )

            return int(row["label"])

        # Modo modelo
        cols = self.feature_columns()

        missing = [
            c
            for c in cols
            if c not in row.index
        ]

        if missing:
            raise ValueError(
                f"Faltan features requeridas por el modelo: {missing}"
            )

        X = pd.DataFrame(
            [row[cols].to_dict()],
            columns=cols,
        )

        return int(self.model.predict(X)[0])


    # ========================================================
    # DIBUJO DEL OJO
    # ========================================================

    def draw_eye(self):

        cx = 380
        cy = 90

        # Título
        self.canvas.create_text(
            cx,
            25,
            text="GESTO OCULAR",
            font=("Arial", 12, "bold"),
        )

        # ----------------------------------------------------
        # BLINK
        # ----------------------------------------------------

        if self.eye_class == 3:

            self.canvas.create_arc(
                cx - 85,
                cy - 30,
                cx + 85,
                cy + 30,
                start=180,
                extent=180,
                style="arc",
                width=5,
            )

            self.canvas.create_line(
                cx - 75,
                cy,
                cx + 75,
                cy,
                width=5,
            )

            self.canvas.create_text(
                cx,
                cy + 48,
                text="BLINK",
                font=("Arial", 11, "bold"),
            )

            return

        # ----------------------------------------------------
        # OJO ABIERTO
        # ----------------------------------------------------

        self.canvas.create_oval(
            cx - 95,
            cy - 42,
            cx + 95,
            cy + 42,
            width=4,
        )

        # Posición iris
        iris_offset = 0

        if self.eye_class == 1:
            iris_offset = 40

        elif self.eye_class == 2:
            iris_offset = 0

        iris_x = cx + iris_offset

        self.canvas.create_oval(
            iris_x - 25,
            cy - 25,
            iris_x + 25,
            cy + 25,
            width=3,
        )

        self.canvas.create_oval(
            iris_x - 9,
            cy - 9,
            iris_x + 9,
            cy + 9,
            fill="black",
        )

        if self.eye_class == 1:

            self.canvas.create_text(
                cx,
                cy + 58,
                text="SACCADE OUTWARD",
                font=("Arial", 11, "bold"),
            )

        elif self.eye_class == 2:

            self.canvas.create_text(
                cx,
                cy + 58,
                text="RETURN SACCADE",
                font=("Arial", 11, "bold"),
            )

        else:

            self.canvas.create_text(
                cx,
                cy + 58,
                text="Esperando señal EOG...",
                font=("Arial", 10),
            )


    # ========================================================
    # TRANSFORMACIÓN GEOMÉTRICA
    # ========================================================

    def rotate_point(self, x, y, cx, cy, angle_deg):

        theta = math.radians(angle_deg)

        xr = (
            cx
            + math.cos(theta) * (x - cx)
            - math.sin(theta) * (y - cy)
        )

        yr = (
            cy
            + math.sin(theta) * (x - cx)
            + math.cos(theta) * (y - cy)
        )

        return xr, yr


    # ========================================================
    # GRIPPER
    # ========================================================

    def draw_gripper(self):

        cx = 380
        cy = 305

        # Texto superior
        self.canvas.create_text(
            cx,
            185,
            text=(
                f"GRIPPER VIRTUAL   |   "
                f"Apertura: {self.opening:.0f}   |   "
                f"Orientación: {self.angle:.0f}°"
            ),
            font=("Arial", 11, "bold"),
        )

        # ----------------------------------------------------
        # ACTUADOR
        # ----------------------------------------------------

        self.canvas.create_rectangle(
            cx - 75,
            cy + 65,
            cx + 75,
            cy + 125,
            width=3,
        )

        self.canvas.create_text(
            cx,
            cy + 95,
            text="ACTUADOR",
            font=("Arial", 12, "bold"),
        )

        # Eje / rotor
        self.canvas.create_oval(
            cx - 18,
            cy + 38,
            cx + 18,
            cy + 74,
            width=3,
        )

        # ----------------------------------------------------
        # GEOMETRÍA BASE DEL GRIPPER
        # ----------------------------------------------------

        left_x = cx - self.opening
        right_x = cx + self.opening

        points = {
            "L1": (cx - 35, cy + 55),
            "L2": (left_x, cy),
            "L3": (left_x, cy - 85),

            "R1": (cx + 35, cy + 55),
            "R2": (right_x, cy),
            "R3": (right_x, cy - 85),
        }

        # Aplicar rotación
        rp = {}

        for name, (x, y) in points.items():

            rp[name] = self.rotate_point(
                x,
                y,
                cx,
                cy,
                self.angle,
            )

        # Brazos
        self.canvas.create_line(
            *rp["L1"],
            *rp["L2"],
            width=9,
        )

        self.canvas.create_line(
            *rp["R1"],
            *rp["R2"],
            width=9,
        )

        # Mordazas
        self.canvas.create_line(
            *rp["L2"],
            *rp["L3"],
            width=15,
        )

        self.canvas.create_line(
            *rp["R2"],
            *rp["R3"],
            width=15,
        )


    # ========================================================
    # ESCENA COMPLETA
    # ========================================================

    def draw_scene(self):

        self.canvas.delete("all")

        self.draw_eye()
        self.draw_gripper()

        # Flecha lógica
        self.canvas.create_text(
            380,
            165,
            text="EOG  →  CLASIFICACIÓN  →  ACCIÓN MECÁNICA",
            font=("Arial", 10, "bold"),
        )


    # ========================================================
    # ANIMACIONES
    # ========================================================

    def animate_opening(self, target, callback=None):

        self.animating = True

        start = self.opening

        delta = (
            target - start
        ) / ANIMATION_STEPS

        delay = (
            ACTION_DURATION_MS
            // ANIMATION_STEPS
        )

        def update(step=0):

            if step >= ANIMATION_STEPS:

                self.opening = target

                self.draw_scene()

                self.animating = False

                if callback:
                    callback()

                return

            self.opening += delta

            self.draw_scene()

            self.root.after(
                delay,
                lambda: update(step + 1),
            )

        update()


    def animate_rotation(self, callback=None):

        self.animating = True

        start = self.angle

        direction = (
            1
            if self.rotation_direction.get() == "HORARIO"
            else -1
        )

        target = start + direction * 90

        delta = (
            target - start
        ) / ANIMATION_STEPS

        delay = (
            ACTION_DURATION_MS
            // ANIMATION_STEPS
        )

        def update(step=0):

            if step >= ANIMATION_STEPS:

                self.angle = target

                self.draw_scene()

                self.animating = False

                if callback:
                    callback()

                return

            self.angle += delta

            self.draw_scene()

            self.root.after(
                delay,
                lambda: update(step + 1),
            )

        update()


    # ========================================================
    # ACCIONES
    # ========================================================

    def apply_action(self, pred, callback=None):

        if pred == 1:

            self.animate_opening(
                125,
                callback,
            )

        elif pred == 2:

            self.animate_opening(
                25,
                callback,
            )

        elif pred == 3:

            self.animate_rotation(
                callback,
            )

        else:

            if callback:
                callback()


    # ========================================================
    # PROCESAMIENTO DE UNA VENTANA
    # ========================================================

    def step(self):

        if self.animating:
            return

        if self.idx >= len(self.df):

            self.gesture_label.config(
                text="FIN DEL STREAM"
            )

            self.running = False

            return

        row = self.df.iloc[self.idx]

        pred = self.predict_row(row)

        truth = (
            int(row["label"])
            if "label" in row.index
            else None
        )

        self.eye_class = pred

        gesture = GESTURE_MAP.get(
            pred,
            f"CLASE {pred}",
        )

        action = ACTION_MAP.get(
            pred,
            "SIN ACCIÓN",
        )

        # ----------------------------------------------------
        # INFORMACIÓN MOSTRADA AL ALUMNO
        # ----------------------------------------------------

        self.gesture_label.config(
            text=(
                f"GESTO EOG: {gesture} "
                f"(Clase {pred})  →  {action}"
            )
        )

        if self.model is None:

            self.prediction_label.config(
                text=(
                    f"Modo demo | "
                    f"Clase de entrada: {pred}"
                )
            )

        else:

            self.prediction_label.config(
                text=(
                    f"Predicción del modelo: {pred}"
                )
            )

        if truth is not None:

            self.truth_label.config(
                text=f"Etiqueta real: {truth}"
            )

            if self.model is not None:

                if pred == truth:

                    self.result_label.config(
                        text="Resultado: CORRECTO"
                    )

                else:

                    self.result_label.config(
                        text="Resultado: ERROR DE CLASIFICACIÓN"
                    )

            else:

                self.result_label.config(
                    text=(
                        "Demo: la etiqueta real "
                        "controla directamente el gripper"
                    )
                )

        else:

            self.truth_label.config(
                text="Etiqueta real: no disponible"
            )

            self.result_label.config(
                text=""
            )

        self.draw_scene()

        self.idx += 1

        # Ejecutar movimiento durante 1 segundo
        self.apply_action(
            pred,
            callback=self.after_action,
        )


    def after_action(self):

        if self.running:

            self.root.after(
                self.interval_ms,
                self.step,
            )


    # ========================================================
    # CONTROLES
    # ========================================================

    def start(self):

        if self.running:
            return

        self.running = True

        self.step()


    def reset(self):

        self.running = False
        self.animating = False

        self.idx = 0

        self.opening = 80.0
        self.angle = 0.0

        self.eye_class = None

        self.gesture_label.config(
            text="GESTO EOG: --"
        )

        self.prediction_label.config(
            text="Predicción: --"
        )

        self.truth_label.config(
            text="Etiqueta real: --"
        )

        self.result_label.config(
            text=""
        )

        self.draw_scene()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="hmi_demo.csv",
        help="CSV utilizado como stream HMI",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Pipeline/modelo .joblib",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=500,
        help="Pausa entre acciones (ms)",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    data_path = Path(args.data)

    if not data_path.exists():

        raise FileNotFoundError(
            f"No existe: {data_path}"
        )

    df = pd.read_csv(data_path)

    # --------------------------------------------------------
    # MODELO
    # --------------------------------------------------------

    model = None

    if args.model:

        model_path = Path(args.model)

        if not model_path.exists():

            raise FileNotFoundError(
                f"No existe: {model_path}"
            )

        model = joblib.load(
            model_path
        )

    # --------------------------------------------------------
    # HMI
    # --------------------------------------------------------

    root = tk.Tk()

    GripperApp(
        root,
        df,
        model=model,
        interval_ms=args.interval,
    )

    root.mainloop()


if __name__ == "__main__":
    main()
