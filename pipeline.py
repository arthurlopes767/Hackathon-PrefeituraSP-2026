"""
pipeline.py — Pré-processamento espacial, cálculo de indicadores,
modelagem (PCA + AHP) e persistência em GeoJSON.

Implementa as camadas descritas na arquitetura:
  - Pré-processamento espacial (delimitação da SMZ via P90 do NDVI + limiar T)
  - Cálculo de indicadores (Blocos 1 e 2 do Apêndice A)
  - Modelagem multicritério (PCA no Bloco 1 -> Índice de Resiliência
    Hidrológica; normalização + ponderação direta no Bloco 2; AHP final)
  - Armazenamento em GeoJSON

O Bloco 3 (Governança) é omitido neste protótipo por falta de acesso aos
shapefiles oficiais (Programa Mananciais / PSA Mananciais / APP-OIDA 2026),
conforme indicado em config.yaml (`governanca.incluir_bloco_3: false`).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def carregar_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalizar_minmax(serie: pd.Series) -> pd.Series:
    """I = (X - Xmin) / (Xmax - Xmin). Retorna 0.5 constante se não houver
    variância (evita divisão por zero em datasets pequenos)."""
    xmin, xmax = serie.min(), serie.max()
    if xmax - xmin == 0:
        return pd.Series(0.5, index=serie.index)
    return (serie - xmin) / (xmax - xmin)


def classificar_padrao(valor_normalizado: float, cfg: dict) -> str:
    faixas = cfg["classificacao"]["normalizado_padrao"]
    if valor_normalizado <= faixas["baixa_max"]:
        return "Baixa"
    if valor_normalizado <= faixas["media_max"]:
        return "Média"
    return "Alta"


def classificar_ndvi_estiagem(valor: float, cfg: dict) -> str:
    faixas = cfg["classificacao"]["ndvi_estiagem"]
    if valor < faixas["baixa_max"]:
        return "Baixa"
    if valor <= faixas["media_max"]:
        return "Média"
    return "Alta"


def classificar_tendencia(beta: float) -> str:
    if beta < 0:
        return "Baixa"
    if beta == 0:
        return "Média"
    return "Alta"


def classificar_amplitude_enso(delta: float) -> str:
    # Direção esperada é negativa: quanto MENOR (mais negativo) o delta,
    # maior a resiliência aparente da vegetação frente ao choque ENSO —
    # portanto a classificação é invertida em relação ao valor bruto.
    if delta > 0:
        return "Baixa"
    if delta == 0:
        return "Média"
    return "Alta"


def classificar_ndvi_p90(valor: float, cfg: dict) -> str:
    faixas = cfg["classificacao"]["ndvi_p90"]
    if faixas["baixa"][0] <= valor <= faixas["baixa"][1]:
        return "Baixa"
    if faixas["media"][0] <= valor <= faixas["media"][1]:
        return "Média"
    if faixas["alta"][0] <= valor <= faixas["alta"][1]:
        return "Alta"
    return "Fora da faixa"


# ---------------------------------------------------------------------------
# Pré-processamento espacial — delimitação da SMZ
# ---------------------------------------------------------------------------

def delimitar_smz(grid_ndvi: list[pd.Series], threshold_T: float) -> dict:
    """Para cada célula de 30 m da grade em torno da nascente, calcula o P90
    do NDVI ao longo da série histórica. Células com P90 > T compõem a SMZ.

    Retorna o P90 médio das células DENTRO da SMZ (usado como o indicador
    NDVIP90 da nascente) e a fração da grade classificada como SMZ.
    """
    p90_por_celula = np.array([serie.quantile(0.90) for serie in grid_ndvi])
    dentro_smz = p90_por_celula > threshold_T

    if dentro_smz.sum() == 0:
        # Nenhuma célula supera o limiar: usa a própria grade completa como
        # fallback, para que a nascente ainda receba um cálculo (evita
        # divisão por zero em datasets sintéticos pequenos).
        dentro_smz = np.ones_like(dentro_smz, dtype=bool)

    return {
        "ndvi_p90_smz": float(p90_por_celula[dentro_smz].mean()),
        "fracao_area_smz": float(dentro_smz.mean()),
        "indices_smz": dentro_smz,
    }


# ---------------------------------------------------------------------------
# Bloco 1 — Vegetação (NDVI)
# ---------------------------------------------------------------------------

def calcular_bloco1(grid_ndvi: list[pd.Series], precipitacao: pd.Series,
                     indices_smz: np.ndarray, cfg: dict) -> dict:
    # Série de NDVI representativa da SMZ: média das células dentro da SMZ
    series_smz = [s for s, dentro in zip(grid_ndvi, indices_smz) if dentro]
    ndvi_medio_serie = pd.concat(series_smz, axis=1).mean(axis=1)

    # --- NDVI médio de estiagem ---
    limite_estiagem = cfg["estiagem"]["precipitacao_limite_mm"]
    meses_estiagem = precipitacao[precipitacao < limite_estiagem].index
    meses_estiagem_validos = ndvi_medio_serie.index.intersection(meses_estiagem)
    ndvi_estiagem = float(ndvi_medio_serie.loc[meses_estiagem_validos].mean()) \
        if len(meses_estiagem_validos) > 0 else float(ndvi_medio_serie.mean())

    # --- Tendência temporal (coeficiente angular) ---
    t = np.arange(len(ndvi_medio_serie))
    beta = float(np.polyfit(t, ndvi_medio_serie.values, 1)[0])

    # --- Amplitude de resposta ENSO (ΔNDVI) ---
    anos_el_nino = set(cfg["enso"]["anos_el_nino"])
    anos_la_nina = set(cfg["enso"]["anos_la_nina"])
    anos_todos = set(ndvi_medio_serie.index.year)
    anos_neutros = anos_todos - anos_el_nino - anos_la_nina

    def media_para_anos(anos):
        mask = ndvi_medio_serie.index.year.isin(anos)
        return ndvi_medio_serie[mask].mean() if mask.any() else np.nan

    ndvi_neutro = media_para_anos(anos_neutros)
    ndvi_enso = media_para_anos(anos_el_nino | anos_la_nina)
    delta_ndvi = float(ndvi_enso - ndvi_neutro) if not np.isnan(ndvi_neutro) else 0.0

    return {
        "ndvi_medio_estiagem": ndvi_estiagem,
        "ndvi_medio_estiagem_classe": classificar_ndvi_estiagem(ndvi_estiagem, cfg),
        "ndvi_tendencia_beta": beta,
        "ndvi_tendencia_classe": classificar_tendencia(beta),
        "ndvi_delta_enso": delta_ndvi,
        "ndvi_delta_enso_classe": classificar_amplitude_enso(delta_ndvi),
    }


# ---------------------------------------------------------------------------
# Bloco 2 — Uso do Solo e Socioeconômico
# ---------------------------------------------------------------------------

def calcular_bloco2_bruto(row: pd.Series) -> dict:
    return {
        "taxa_impermeabilizacao_pct": row["taxa_impermeabilizacao_pct"],
        "adequacao_app_pct": row["adequacao_app_pct"],
        "distancia_urbana_m": row["distancia_urbana_m"],
        "ipvs_categoria": row["ipvs_categoria"],
    }


# ---------------------------------------------------------------------------
# Orquestração completa
# ---------------------------------------------------------------------------

def rodar_pipeline(df_nascentes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    T = cfg["smz"]["ndvi_threshold_T"]

    registros = []
    for _, row in df_nascentes.iterrows():
        smz = delimitar_smz(row["grid_ndvi"], T)
        bloco1 = calcular_bloco1(row["grid_ndvi"], row["precipitacao_mensal"],
                                  smz["indices_smz"], cfg)
        bloco2_bruto = calcular_bloco2_bruto(row)

        registros.append({
            "id_nascente": row["id_nascente"],
            "cluster_subbacia": row["cluster_subbacia"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "fracao_area_smz": smz["fracao_area_smz"],
            "ndvi_p90_smz": smz["ndvi_p90_smz"],
            "ndvi_p90_classe": classificar_ndvi_p90(smz["ndvi_p90_smz"], cfg),
            **bloco1,
            **bloco2_bruto,
        })

    df = pd.DataFrame(registros)

    # --- Normalização min-max do Bloco 2 (com correção de direção) ---------
    df["norm_impermeabilizacao"] = 1 - normalizar_minmax(df["taxa_impermeabilizacao_pct"])
    df["norm_app"] = normalizar_minmax(df["adequacao_app_pct"])
    df["norm_distancia_urbana"] = normalizar_minmax(df["distancia_urbana_m"])
    df["norm_ipvs"] = normalizar_minmax(df["ipvs_categoria"])

    for col_norm, col_saida in [
        ("norm_impermeabilizacao", "classe_impermeabilizacao"),
        ("norm_app", "classe_app"),
        ("norm_distancia_urbana", "classe_distancia_urbana"),
        ("norm_ipvs", "classe_ipvs"),
    ]:
        df[col_saida] = df[col_norm].apply(lambda v: classificar_padrao(v, cfg))

    # --- PCA no Bloco 1 -> Índice de Resiliência Hidrológica ---------------
    cols_bloco1 = ["ndvi_medio_estiagem", "ndvi_tendencia_beta", "ndvi_delta_enso"]
    X1 = df[cols_bloco1].copy()
    # Inverte o sinal do delta ENSO antes do PCA (direção esperada é
    # negativa, mas queremos que "mais resiliente" = valor maior em todos os
    # componentes de entrada).
    X1["ndvi_delta_enso"] = -X1["ndvi_delta_enso"]
    X1_std = (X1 - X1.mean()) / X1.std().replace(0, 1)

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X1_std.values).flatten()
    # Garante que PC1 seja positivamente correlacionado com o NDVI de
    # estiagem, para que "maior PC1 = maior resiliência".
    if np.corrcoef(pc1, X1_std["ndvi_medio_estiagem"])[0, 1] < 0:
        pc1 = -pc1
    df["pca_variancia_explicada"] = float(pca.explained_variance_ratio_[0])
    df["indice_resiliencia_hidrologica"] = normalizar_minmax(pd.Series(pc1))

    # --- Bloco 2 -> ponderação direta (pesos iguais, ajustável em config) --
    df["indice_uso_solo_socioeconomico"] = df[[
        "norm_impermeabilizacao", "norm_app", "norm_distancia_urbana", "norm_ipvs"
    ]].mean(axis=1)

    return df


def aplicar_ahp(df: pd.DataFrame, peso_resiliencia: float,
                 peso_uso_solo: float) -> pd.DataFrame:
    """Combina os blocos via AHP. Bloco 3 (Governança) está desativado neste
    protótipo, então a soma dos pesos usados é normalizada entre os dois
    blocos disponíveis."""
    soma = peso_resiliencia + peso_uso_solo
    if soma == 0:
        peso_resiliencia = peso_uso_solo = 0.5
        soma = 1.0
    w_res = peso_resiliencia / soma
    w_uso = peso_uso_solo / soma

    df = df.copy()
    df["indice_prioridade_recuperacao"] = (
        w_res * df["indice_resiliencia_hidrologica"] +
        w_uso * df["indice_uso_solo_socioeconomico"]
    )
    return df


# ---------------------------------------------------------------------------
# Armazenamento — GeoJSON
# ---------------------------------------------------------------------------

def exportar_geojson(df: pd.DataFrame) -> dict:
    """Monta um FeatureCollection GeoJSON (geometria Point) sem depender de
    geopandas, para manter o protótipo leve — compatível com o escopo do
    hackathon."""
    colunas_excluir = {"latitude", "longitude"}
    features = []
    for _, row in df.iterrows():
        props = {k: (v if not isinstance(v, (np.floating,)) else float(v))
                  for k, v in row.items() if k not in colunas_excluir}
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["longitude"], row["latitude"]],
            },
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}
