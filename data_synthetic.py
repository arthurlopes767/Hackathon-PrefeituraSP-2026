"""
Camada de Coleta de Dados 
=======================================================================

IMPORTANTE — LEIA ANTES DE USAR EM PRODUÇÃO
-----------------------------------------------------------------------
Este módulo NÃO acessa o DataGEO, o Google Earth Engine, o INMET, o
MapBiomas, o GeoSampa ou a Fundação SEADE. Essas integrações exigem
credenciais e/ou downloads de bases que não estão disponíveis no 
ambiente onde este protótipo foi construído.

Para permitir que TODO o restante do pipeline (pré-processamento espacial,
cálculo de indicadores, PCA, AHP, dashboard) seja demonstrado de ponta a
ponta, este módulo gera um conjunto de nascentes sintéticas com a MESMA
ESTRUTURA que os dados reais teriam:

  - pontos de nascente distribuídos dentro do bounding box do município de
    São Paulo;
  - clusterização por "sub-bacia" (simulada via k-means sobre as
    coordenadas, no lugar da malha real de sub-bacias hidrográficas);
  - uma grade de células de 30 m ao redor de cada nascente (substituindo o
    raster NDVI do Landsat via GEE), cada uma com série temporal mensal de
    NDVI (2014-2025) com sazonalidade, tendência e resposta a El Niño/La Niña
    simuladas;
  - série de precipitação mensal sintética (substituindo o INMET);
  - taxa de impermeabilização do entorno, adequação da APP, distância à
    mancha urbana e IPVS sintéticos (substituindo MapBiomas/GeoSampa/SEADE).

>>> PARA IR A PRODUÇÃO: substitua as funções deste arquivo por chamadas
    reais (ee.ImageCollection para o NDVI via GEE, leitura dos shapefiles do
    DataGEO/GeoSampa via geopandas, API do INMET, tabela do IPVS da SEADE
    etc.), mantendo a mesma assinatura de saída (mesmas colunas), para que
    `pipeline.py` e `app.py` continuem funcionando sem alterações.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Bounding box aproximado do município de São Paulo
SP_LAT_MIN, SP_LAT_MAX = -23.78, -23.36
SP_LON_MIN, SP_LON_MAX = -46.83, -46.36

MESES = pd.date_range("2014-01-01", "2025-12-01", freq="MS")
N_CELULAS_GRID = 5  # grade NxN de células de 30 m ao redor de cada nascente


def _gerar_serie_ndvi(rng: np.random.Generator, base: float, tendencia: float,
                       amplitude_sazonal: float, choque_enso: dict) -> pd.Series:
    """Gera uma série temporal mensal de NDVI sintética com sazonalidade,
    tendência linear e resposta a anos de El Niño/La Niña."""
    t = np.arange(len(MESES))
    sazonal = amplitude_sazonal * np.sin(2 * np.pi * (t % 12) / 12 + np.pi / 6)
    tend = tendencia * (t / 12)  # tendência por ano
    ruido = rng.normal(0, 0.02, size=len(MESES))

    enso_efeito = np.zeros(len(MESES))
    for i, data in enumerate(MESES):
        ano = data.year
        if ano in choque_enso.get("el_nino", []):
            enso_efeito[i] = choque_enso.get("efeito_el_nino", 0.0)
        elif ano in choque_enso.get("la_nina", []):
            enso_efeito[i] = choque_enso.get("efeito_la_nina", 0.0)

    serie = base + sazonal + tend + enso_efeito + ruido
    return pd.Series(np.clip(serie, -1, 1), index=MESES)


def _gerar_precipitacao(rng: np.random.Generator) -> pd.Series:
    """Precipitação mensal sintética com sazonalidade (verão chuvoso /
    inverno seco), aproximando o regime climático de São Paulo."""
    t = np.arange(len(MESES))
    base = 90 + 80 * np.sin(2 * np.pi * (t % 12) / 12 - np.pi / 2)
    ruido = rng.normal(0, 20, size=len(MESES))
    return pd.Series(np.clip(base + ruido, 0, None), index=MESES)


def gerar_nascentes_sinteticas(n: int = 60, seed: int = 42,
                                n_clusters: int = 6) -> pd.DataFrame:
    """Gera o dataset sintético de nascentes pontuais + suas grades de
    células NDVI + atributos de uso do solo/socioeconômicos.

    Retorna um DataFrame com uma linha por nascente e, na coluna
    `grid_ndvi`, uma lista de `N_CELULAS_GRID**2` séries temporais de NDVI
    (uma por célula de 30 m do entorno) — usada na etapa de delimitação da
    SMZ.
    """
    rng = np.random.default_rng(seed)

    lats = rng.uniform(SP_LAT_MIN, SP_LAT_MAX, n)
    lons = rng.uniform(SP_LON_MIN, SP_LON_MAX, n)

    # Clusterização por sub-bacia: substitui a malha real de sub-bacias
    # hidrográficas por um agrupamento espacial (k-means) das coordenadas.
    from sklearn.cluster import KMeans
    coords = np.column_stack([lats, lons])
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    clusters = kmeans.fit_predict(coords)

    # Cada cluster/sub-bacia tem um "clima base" próprio (correlação
    # espacial), para que nascentes vizinhas tenham NDVI parecido.
    cluster_base = {c: rng.uniform(0.30, 0.62) for c in range(n_clusters)}
    cluster_tendencia = {c: rng.uniform(-0.03, 0.03) for c in range(n_clusters)}
    cluster_efeito_nino = {c: rng.uniform(-0.06, -0.01) for c in range(n_clusters)}
    cluster_efeito_nina = {c: rng.uniform(0.01, 0.05) for c in range(n_clusters)}

    registros = []
    for i in range(n):
        cluster = int(clusters[i])
        base = cluster_base[cluster] + rng.normal(0, 0.03)
        tendencia = cluster_tendencia[cluster] + rng.normal(0, 0.01)
        choque_enso = {
            "el_nino": [2014, 2015, 2018, 2023],
            "la_nina": [2016, 2017, 2020, 2021, 2022],
            "efeito_el_nino": cluster_efeito_nino[cluster],
            "efeito_la_nina": cluster_efeito_nina[cluster],
        }

        # Grade de N x N células de 30 m ao redor da nascente, cada uma com
        # pequena variação em torno do sinal do cluster (heterogeneidade
        # intra-SMZ).
        grid_series = []
        for _cel in range(N_CELULAS_GRID ** 2):
            base_celula = base + rng.normal(0, 0.05)
            grid_series.append(
                _gerar_serie_ndvi(rng, base_celula, tendencia, 0.04, choque_enso)
            )

        precip = _gerar_precipitacao(rng)

        # Uso do solo / socioeconômico (substituem MapBiomas/GeoSampa/SEADE)
        taxa_impermeabilizacao_pct = float(np.clip(rng.normal(35, 20), 0, 100))
        adequacao_app_pct = float(np.clip(rng.normal(55, 25), 0, 100))
        distancia_urbana_m = float(np.clip(rng.exponential(400), 0, 3000))
        ipvs_categoria = int(rng.integers(1, 7))  # 1 (baixíssima) a 6 (altíssima)

        registros.append({
            "id_nascente": f"NASC_{i:03d}",
            "cluster_subbacia": cluster,
            "latitude": lats[i],
            "longitude": lons[i],
            "grid_ndvi": grid_series,
            "precipitacao_mensal": precip,
            "taxa_impermeabilizacao_pct": taxa_impermeabilizacao_pct,
            "adequacao_app_pct": adequacao_app_pct,
            "distancia_urbana_m": distancia_urbana_m,
            "ipvs_categoria": ipvs_categoria,
        })

    return pd.DataFrame(registros)
