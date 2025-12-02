# ============================================================
# app.py — Tablero Airbnb amigable para viajeros (clusters + filtros)
# Proyecto Global Classroom - Etapa 5
# ============================================================

import pathlib

import numpy as np
import pandas as pd

import dash
from dash import html, dcc, Input, Output
import plotly.express as px

import tensorflow as tf

# ------------------------------------------------------------
# 1. RUTAS Y ARCHIVOS
# ------------------------------------------------------------
PATH = pathlib.Path(__file__).parent
CSV_PATH = PATH.joinpath("listings_model_no_outliers.csv")
MODEL_PATH = PATH.joinpath("nn_regression_price.h5")

# ------------------------------------------------------------
# 2. CARGA DEL DATASET LIMPIO Y CONSTRUCCIÓN DE FEATURES
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

print("Shape del DataFrame:", df.shape)
print("Primeras columnas:", df.columns.tolist()[:20])

# Detectar columna de precio
posibles_targets = ["price", "price_night", "price_per_night"]
target_col = None
for col in posibles_targets:
    if col in df.columns:
        target_col = col
        break
if target_col is None:
    raise ValueError("No se encontró ninguna columna de precio en el DataFrame.")

print(f"Usaremos la columna '{target_col}' como variable objetivo (y).")

# Convertir precio a numérico si viene como texto
if df[target_col].dtype == "O":
    df[target_col] = (
        df[target_col]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .astype(float)
    )

# Eliminar filas sin precio
df = df[df[target_col].notna()].copy()

# Construcción de X (todas las numéricas menos las que descartamos)
cols_descartar = [
    target_col,
    "id",
    "listing_id",
    "name",
    "description",
    "host_name",
    "host_id",
    "last_review",
]
cols_descartar = [c for c in cols_descartar if c in df.columns]

X_raw = df.drop(columns=cols_descartar)
X_num = X_raw.select_dtypes(include=[np.number]).copy()
print("Variables numéricas iniciales:", X_num.shape[1])

# Columnas sin variación
cols_sin_variacion = X_num.columns[X_num.nunique() <= 1].tolist()
print("Columnas sin variación (se eliminan):", len(cols_sin_variacion))

# Dummies casi constantes
binary_cols = []
for col in X_num.columns:
    valores = set(X_num[col].dropna().unique())
    if valores.issubset({0, 1}):
        binary_cols.append(col)

proporcion_positivos = X_num[binary_cols].mean()
umbral = 0.01
cols_casi_todo_cero = proporcion_positivos[proporcion_positivos < umbral].index.tolist()
cols_casi_todo_uno = proporcion_positivos[proporcion_positivos > (1 - umbral)].index.tolist()

print("Dummies casi siempre 0 (se eliminan):", len(cols_casi_todo_cero))
print("Dummies casi siempre 1 (se eliminan):", len(cols_casi_todo_uno))

cols_eliminar_por_poca_info = list(
    set(cols_sin_variacion + cols_casi_todo_cero + cols_casi_todo_uno)
)

X = X_num.drop(columns=cols_eliminar_por_poca_info).copy()
feature_names = X.columns.tolist()
print("Variables finales usadas como features:", len(feature_names))

y = df[target_col].values

# ------------------------------------------------------------
# 2.1. COLUMNA "RECOMMENDED"
#      Definición: rating >= 4.0, cleanliness >= 4.0 y >= 5 reseñas
# ------------------------------------------------------------
if (
    "review_scores_rating" in df.columns
    and "review_scores_cleanliness" in df.columns
    and "number_of_reviews" in df.columns
):
    df["recommended"] = (
        (df["review_scores_rating"] >= 4.0)
        & (df["review_scores_cleanliness"] >= 4.0)
        & (df["number_of_reviews"] >= 5)
    )
else:
    df["recommended"] = False

# ------------------------------------------------------------
# 3. DATAFRAME PARA EL MAPA (VISTA “VIAJERO”)
# ------------------------------------------------------------
cols_mapa = [
    "latitude",
    "longitude",
    "accommodates",
    "bedrooms",
    "beds",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",        # para el slider de rating
    "review_scores_cleanliness",   # para definir "recommended"
    target_col,
    "name",
    "recommended",
]
cols_mapa = [c for c in cols_mapa if c in df.columns]

df_map = df[cols_mapa].dropna(subset=["latitude", "longitude"]).copy()
df_map["row_id"] = df_map.index  # conecta con df y X

# Crear una “zona” aproximada para filtrar menos puntos en el mapa
def asignar_zona(lon):
    if lon <= -0.25:
        return "Oeste"
    elif lon >= 0.0:
        return "Este"
    else:
        return "Centro"

df_map["zona"] = df_map["longitude"].apply(asignar_zona)

# ------------------------------------------------------------
# 4. CARGA DEL MODELO DE REDES NEURONALES (INFERENCIA)
# ------------------------------------------------------------
print(f"Cargando modelo desde: {MODEL_PATH}")
nn_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ------------------------------------------------------------
# 5. CONFIGURACIÓN DE DASH Y ESTILOS
# ------------------------------------------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Airbnb para Viajeros – Regresión NN"
server = app.server

BACKGROUND_COLOR = "#f4f5fb"
CARD_BG = "#ffffff"
CARD_SHADOW = "0 4px 12px rgba(0, 0, 0, 0.06)"

def pretty_card(children, style_extra=None):
    """Caja blanca con bordes redondeados y sombra suave."""
    base_style = {
        "backgroundColor": CARD_BG,
        "borderRadius": "16px",
        "padding": "18px",
        "boxShadow": CARD_SHADOW,
    }
    if style_extra:
        base_style.update(style_extra)
    return html.Div(children, style=base_style)

def stat_card(titulo, valor, subtitulo):
    """Tarjeta pequeña de métrica para el viajero."""
    return pretty_card(
        [
            html.H6(titulo, style={"marginBottom": "2px"}),
            html.H4(
                valor,
                style={
                    "marginTop": "0px",
                    "marginBottom": "2px",
                    "fontSize": "20px",
                    "fontWeight": "600",
                },
            ),
            html.P(
                subtitulo,
                style={"fontSize": "11px", "color": "#777", "marginTop": "0px"},
            ),
        ],
        style_extra={
            "flex": "1 1 22%",
            "minWidth": "140px",
            "padding": "12px",
        },
    )

# ------------------------------------------------------------
# 6. LAYOUT ÚNICO (SIN PESTAÑAS)
# ------------------------------------------------------------
reg_nn_layout = html.Div(
    [
        # Título de la pestaña/vista
        html.Div(
            [
                html.H2(
                    "Explora alojamientos en Londres",
                    style={"marginBottom": "4px"},
                ),
                html.P(
                    "Elige tus preferencias, explora el mapa y mira cuántos alojamientos "
                    "recomendados encuentras según su calidad y reseñas.",
                    style={"color": "#555", "marginTop": "0px"},
                ),
            ],
            style={"marginBottom": "16px"},
        ),

        # TARJETAS ARRIBA (resumen rápido)
        html.Div(
            id="summary-row",
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "20px",
                "marginBottom": "20px",
            },
        ),

        # Fila central: filtros + mapa
        html.Div(
            [
                # Columna izquierda: filtros
                html.Div(
                    pretty_card(
                        [
                            html.H4("1. Personaliza tu estadía 👇"),

                            html.Label("Zona de la ciudad"),
                            dcc.Dropdown(
                                id="dropdown-zone",
                                options=[
                                    {"label": "Toda la ciudad", "value": "all"},
                                    {"label": "Zona oeste", "value": "Oeste"},
                                    {"label": "Zona centro", "value": "Centro"},
                                    {"label": "Zona este", "value": "Este"},
                                ],
                                value="all",
                                clearable=False,
                                style={"marginBottom": "14px"},
                            ),

                            html.Label("Número de huéspedes"),
                            dcc.Dropdown(
                                id="dropdown-guests",
                                options=[
                                    {
                                        "label": f"{i} huésped" if i == 1 else f"{i} huéspedes",
                                        "value": i,
                                    }
                                    for i in range(1, 8)
                                ],
                                value=2,
                                clearable=False,
                                style={"marginBottom": "14px"},
                            ),

                            html.Label("Número de noches"),
                            html.Div(
                                dcc.Slider(
                                    id="slider-nights",
                                    min=1,
                                    max=14,
                                    step=1,
                                    value=3,
                                    marks={1: "1", 3: "3", 7: "7", 14: "14"},
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                                style={"marginBottom": "18px"},
                            ),

                            html.Label("Mínimo de reseñas"),
                            html.Div(
                                dcc.Slider(
                                    id="slider-min-reviews",
                                    min=0,
                                    max=100,
                                    step=5,
                                    value=10,
                                    marks={0: "0", 25: "25", 50: "50", 100: "100+"},
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": False,
                                    },
                                ),
                                style={"marginBottom": "18px"},
                            ),

                            html.Label("Rating mínimo (0–5)"),
                            html.Div(
                                dcc.Slider(
                                    id="slider-rating",
                                    min=0,
                                    max=5,
                                    step=0.5,
                                    value=3.5,
                                    marks={0: "0", 2: "2", 3: "3", 4: "4", 5: "5"},
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": False,
                                    },
                                ),
                                style={"marginBottom": "12px"},
                            ),

                            html.P(
                                "Usamos estos filtros para mostrarte en el mapa "
                                "los alojamientos que más se parecen a lo que buscas.",
                                style={"fontSize": "12px", "color": "#777"},
                            ),
                        ]
                    ),
                    style={"flex": "0 0 28%", "minWidth": "260px"},
                ),

                # Columna derecha: solo mapa
                html.Div(
                    [
                        pretty_card(
                            [
                                html.H4("2. Explora el mapa 🗺️"),
                                html.P(
                                    "Cada círculo agrupa alojamientos cercanos. "
                                    "El tamaño indica cuántos hay en la zona. "
                                    "Pasa el cursor para ver el precio promedio "
                                    "y qué porcentaje son 'recomendados'.",
                                    style={"fontSize": "13px", "color": "#666"},
                                ),
                                dcc.Graph(
                                    id="map-figure",
                                    config={"displayModeBar": False},
                                    style={"height": "430px"},
                                ),
                            ],
                            style_extra={"marginBottom": "18px"},
                        ),
                    ],
                    style={"flex": "0 0 70%", "minWidth": "320px"},
                ),
            ],
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "20px",
                "marginBottom": "20px",
            },
        ),
    ],
    style={
        "padding": "24px",
        "backgroundColor": BACKGROUND_COLOR,
        "minHeight": "100vh",
        "fontFamily": '"Helvetica Neue", Arial, sans-serif',
        "color": "#222",
    },
)

# Layout general SIN pestañas
app.layout = html.Div(
    [
        html.H1(
            "Tablero Airbnb – Proyecto Global Classroom",
            style={
                "textAlign": "center",
                "marginBottom": "10px",
                "marginTop": "16px",
                "fontFamily": '"Helvetica Neue", Arial, sans-serif',
            },
        ),
        html.Hr(style={"marginBottom": "0px"}),
        reg_nn_layout,
    ],
    style={"backgroundColor": BACKGROUND_COLOR, "minHeight": "100vh"},
)

# ------------------------------------------------------------
# 7. CALLBACKS
# ------------------------------------------------------------

# Filtros -> mapa (con clusters) y tarjetas de resumen
@app.callback(
    [
        Output("map-figure", "figure"),
        Output("summary-row", "children"),
    ],
    [
        Input("dropdown-zone", "value"),
        Input("dropdown-guests", "value"),
        Input("slider-nights", "value"),
        Input("slider-min-reviews", "value"),
        Input("slider-rating", "value"),
    ],
)
def update_exploration(zone, guests, nights, min_reviews, min_rating):
    # Partimos de todos los listings
    dff = df_map.copy()

    # Filtro por zona de la ciudad
    if zone is not None and zone != "all":
        dff = dff[dff["zona"] == zone]

    # Filtro por número de huéspedes
    if guests is not None and "accommodates" in dff.columns:
        dff = dff[dff["accommodates"] >= guests]

    # Filtro por noches mínimas
    if nights is not None and "minimum_nights" in dff.columns:
        dff = dff[dff["minimum_nights"] <= nights]

    # Filtro por número de reseñas
    if min_reviews is not None and "number_of_reviews" in dff.columns:
        dff = dff[dff["number_of_reviews"] >= min_reviews]

    # Filtro por rating mínimo
    if min_rating is not None and "review_scores_rating" in dff.columns:
        dff = dff[dff["review_scores_rating"] >= min_rating]

    # Si no hay nada, devolvemos cosas vacías amables
    if len(dff) == 0:
        empty_map = px.scatter_mapbox()
        empty_map.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=0, b=0),
            height=430,
        )
        summary_cards = [
            stat_card("0 alojamientos", "—", "Ajusta los filtros para ver opciones.")
        ]
        return empty_map, summary_cards

    # --- CLUSTER GEO (aproximado): agrupamos por lat/lon redondeada ---
    dff_cluster = dff.copy()
    dff_cluster["lat_cluster"] = dff_cluster["latitude"].round(2)
    dff_cluster["lon_cluster"] = dff_cluster["longitude"].round(2)

    agg_dict = {
        "latitude": ("latitude", "mean"),
        "longitude": ("longitude", "mean"),
        "avg_price": (target_col, "mean"),
        "avg_reviews": ("number_of_reviews", "mean"),
        "avg_accommodates": ("accommodates", "mean"),
        "count_listings": ("row_id", "count"),
    }
    if "review_scores_rating" in dff_cluster.columns:
        agg_dict["avg_rating"] = ("review_scores_rating", "mean")
    if "recommended" in dff_cluster.columns:
        # proporción de alojamientos recomendados en cada cluster
        agg_dict["prop_recommended"] = ("recommended", "mean")
    if "name" in dff_cluster.columns:
        agg_dict["name"] = ("name", "first")

    clusters = (
        dff_cluster.groupby(["lat_cluster", "lon_cluster"])
        .agg(**agg_dict)
        .reset_index(drop=True)
    )

    hover_name_col = "name" if "name" in clusters.columns else None

    hover_data_cols = {}
    for col in [
        "avg_price",
        "avg_reviews",
        "avg_accommodates",
        "avg_rating",
        "count_listings",
        "prop_recommended",
    ]:
        if col in clusters.columns:
            hover_data_cols[col] = True

    fig_map = px.scatter_mapbox(
        clusters,
        lat="latitude",
        lon="longitude",
        color="avg_price",
        size="count_listings",
        hover_name=hover_name_col,
        hover_data=hover_data_cols,
        zoom=10,
    )
    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        height=430,
    )

    # TARJETAS RESUMEN (nivel listing, no cluster)
    n_listings = len(dff)
    price_mean = dff[target_col].mean() if target_col in dff.columns else np.nan
    reviews_mean = (
        dff["number_of_reviews"].mean()
        if "number_of_reviews" in dff.columns
        else np.nan
    )
    rating_mean = (
        dff["review_scores_rating"].mean()
        if "review_scores_rating" in dff.columns
        else np.nan
    )

    # Recomendados
    if "recommended" in dff.columns:
        n_recommended = int(dff["recommended"].sum())
        pct_recommended = (n_recommended / n_listings) * 100 if n_listings > 0 else 0
    else:
        n_recommended = 0
        pct_recommended = 0.0

    summary_cards = [
        stat_card(
            "Alojamientos disponibles",
            f"{n_listings}",
            "Que cumplen tus filtros actuales.",
        ),
        stat_card(
            "Precio promedio por noche",
            f"{price_mean:.2f}" if not np.isnan(price_mean) else "N/A",
            "Referencia rápida para tu viaje.",
        ),
        stat_card(
            "Alojamientos recomendados",
            f"{n_recommended}",
            "Rating ≥ 4.0, limpieza ≥ 4.0 y ≥ 5 reseñas.",
        ),
        stat_card(
            "Rating promedio",
            f"{rating_mean:.1f}" if not np.isnan(rating_mean) else "N/A",
            "Calidad general según otros viajeros.",
        ),
    ]

    return fig_map, summary_cards


# ------------------------------------------------------------
# 8. MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
