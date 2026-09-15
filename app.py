"""
app.py — Dashboard Streamlit (Camada de Visualização)
=========================================================================
Sistema de Apoio à Decisão para Priorização da Recuperação de Nascentes
Urbanas frente às Mudanças Climáticas — PROTÓTIPO (Hackathon Climático
2026: Água).

Lê/gera os dados via `data_synthetic.py` (ver aviso no topo daquele
arquivo sobre a substituição das fontes oficiais), roda o pipeline em
`pipeline.py` (SMZ -> indicadores -> PCA -> AHP) e exibe dois mapas
interativos (Folium) com sliders para a Prefeitura simular cenários de
priorização ajustando os pesos do AHP ao vivo.

Rodar com:  streamlit run app.py
"""

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import data_synthetic as ds
import pipeline as pl

st.set_page_config(
    page_title="Priorização de Nascentes — SP",
    page_icon="💧",
    layout="wide",
)

CONFIG_PATH = "config.yaml"


# ---------------------------------------------------------------------------
# Cache de dados / pipeline
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def carregar_config():
    return pl.carregar_config(CONFIG_PATH)


@st.cache_data(show_spinner="Gerando dataset de nascentes...")
def gerar_dados(n_nascentes: int, seed: int, n_clusters: int):
    return ds.gerar_nascentes_sinteticas(n=n_nascentes, seed=seed, n_clusters=n_clusters)


@st.cache_data(show_spinner="Rodando pipeline: SMZ, indicadores e PCA...")
def rodar_pipeline_cache(df_nascentes: pd.DataFrame, _cfg: dict):
    return pl.rodar_pipeline(df_nascentes, _cfg)


# ---------------------------------------------------------------------------
# Sidebar — parâmetros do cenário
# ---------------------------------------------------------------------------

cfg = carregar_config()

st.sidebar.title("💧 Painel de Simulação")

st.sidebar.markdown("### Dados de entrada")
st.sidebar.caption(
    "⚠️ Protótipo: nascentes, séries de NDVI, precipitação, uso do solo e "
    "IPVS são **sintéticos**, pois este ambiente não tem acesso ao DataGEO, "
    "Google Earth Engine, INMET, MapBiomas, GeoSampa ou SEADE. A estrutura "
    "do cálculo é idêntica à que seria usada com dados reais — basta trocar "
    "`data_synthetic.py` por conectores reais."
)
n_nascentes = st.sidebar.slider("Número de nascentes simuladas", 10, 150, 60, step=10)
seed = st.sidebar.number_input("Semente aleatória (seed)", value=42, step=1)
n_clusters = st.sidebar.slider("Nº de sub-bacias (clusters)", 2, 12, 6)

st.sidebar.markdown("---")
st.sidebar.markdown("### Pesos do AHP")
st.sidebar.caption(
    "O Bloco 3 (Governança) está **desativado neste protótipo** — sem acesso "
    "aos shapefiles do Programa Mananciais / PSA Mananciais / APP-OIDA 2026. "
    "O Índice de Prioridade combina apenas Resiliência Hidrológica e Uso do "
    "Solo/Socioeconômico."
)
peso_resiliencia = st.sidebar.slider(
    "Resiliência Hidrológica (Bloco 1)", 0.0, 1.0,
    float(cfg["ahp"]["pesos_default"]["resiliencia_hidrologica"]), step=0.05,
)
peso_uso_solo = st.sidebar.slider(
    "Uso do Solo e Socioeconômico (Bloco 2)", 0.0, 1.0,
    float(cfg["ahp"]["pesos_default"]["uso_solo_socioeconomico"]), step=0.05,
)
st.sidebar.caption(f"Governança (Bloco 3): fixo em 0.00 (desativado)")

soma_pesos = peso_resiliencia + peso_uso_solo
if soma_pesos == 0:
    st.sidebar.error("A soma dos pesos não pode ser zero. Ajuste os sliders.")
else:
    st.sidebar.caption(
        f"Pesos normalizados aplicados: Resiliência = "
        f"{peso_resiliencia/soma_pesos:.2f} · Uso do Solo = "
        f"{peso_uso_solo/soma_pesos:.2f}"
    )

st.sidebar.markdown("---")
limiar_T = st.sidebar.slider(
    "Limiar T (P90 NDVI) para delimitar a SMZ", 0.30, 0.70,
    float(cfg["smz"]["ndvi_threshold_T"]), step=0.01,
    help="Só reflete no cálculo após clicar em 'Atualizar análise'.",
)
atualizar = st.sidebar.button("🔄 Atualizar análise", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Execução do pipeline
# ---------------------------------------------------------------------------

df_raw = gerar_dados(n_nascentes, seed, n_clusters)

cfg_ajustado = dict(cfg)
cfg_ajustado["smz"] = dict(cfg["smz"])
cfg_ajustado["smz"]["ndvi_threshold_T"] = limiar_T

df_indicadores = rodar_pipeline_cache(df_raw, cfg_ajustado)
df_final = pl.aplicar_ahp(df_indicadores, peso_resiliencia, peso_uso_solo)


# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------

st.title("💧 Priorização da Recuperação de Nascentes Urbanas — São Paulo")
st.markdown(
    "Sistema de apoio à decisão baseado em análise multicritério "
    "(PCA + AHP) para identificar o potencial de refúgio hidrológico das "
    "nascentes e priorizar ações de recuperação, adaptado de "
    "**Cartwright & Johnson (2018)**."
)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Nascentes analisadas", len(df_final))
col_b.metric("Sub-bacias (clusters)", df_final["cluster_subbacia"].nunique())
col_c.metric("Índice de Resiliência (médio)", f"{df_final['indice_resiliencia_hidrologica'].mean():.2f}")
col_d.metric("Índice de Prioridade (médio)", f"{df_final['indice_prioridade_recuperacao'].mean():.2f}")

st.markdown("---")


# ---------------------------------------------------------------------------
# Mapas
# ---------------------------------------------------------------------------

def cor_por_indice(valor: float) -> str:
    if valor >= 0.66:
        return "#1a9850"
    if valor >= 0.33:
        return "#fee08b"
    return "#d73027"


def montar_mapa(df: pd.DataFrame, coluna_indice: str, titulo_popup: str) -> folium.Map:
    centro = [df["latitude"].mean(), df["longitude"].mean()]
    mapa = folium.Map(location=centro, zoom_start=10, tiles="CartoDB positron")

    for _, row in df.iterrows():
        valor = row[coluna_indice]
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6 + 6 * valor,
            color=cor_por_indice(valor),
            fill=True,
            fill_color=cor_por_indice(valor),
            fill_opacity=0.8,
            weight=1,
            popup=folium.Popup(
                f"<b>{row['id_nascente']}</b><br>"
                f"{titulo_popup}: {valor:.2f}<br>"
                f"Sub-bacia: {int(row['cluster_subbacia'])}<br>"
                f"NDVI P90 (SMZ): {row['ndvi_p90_smz']:.2f} ({row['ndvi_p90_classe']})<br>"
                f"NDVI estiagem: {row['ndvi_medio_estiagem_classe']}<br>"
                f"Impermeabilização: {row['classe_impermeabilizacao']}<br>"
                f"IPVS: {row['classe_ipvs']}",
                max_width=280,
            ),
        ).add_to(mapa)

    legenda_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 6px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: 13px;">
        <b>Legenda</b><br>
        <span style="color:#1a9850;">●</span> Alta (≥ 0.66)<br>
        <span style="color:#fee08b;">●</span> Média (0.33–0.66)<br>
        <span style="color:#d73027;">●</span> Baixa (&lt; 0.33)
    </div>
    """
    mapa.get_root().html.add_child(folium.Element(legenda_html))
    return mapa


tab_resiliencia, tab_prioridade, tab_tabela = st.tabs([
    "🌱 Mapa de Resiliência Hidrológica",
    "🎯 Mapa de Prioridade de Recuperação",
    "📋 Tabela de indicadores",
])

with tab_resiliencia:
    st.caption(
        "Potencial de cada nascente atuar como **refúgio hidrológico** "
        "frente às mudanças climáticas (Bloco 1, síntese por PCA)."
    )
    mapa1 = montar_mapa(df_final, "indice_resiliencia_hidrologica", "Resiliência")
    st_folium(mapa1, width=None, height=560, key="mapa_resiliencia")

with tab_prioridade:
    st.caption(
        "Decisão final de priorização para ações de recuperação/conservação "
        "(AHP sobre os Blocos 1 e 2, com os pesos definidos na barra lateral)."
    )
    mapa2 = montar_mapa(df_final, "indice_prioridade_recuperacao", "Prioridade")
    st_folium(mapa2, width=None, height=560, key="mapa_prioridade")

with tab_tabela:
    colunas_exibir = [
        "id_nascente", "cluster_subbacia",
        "ndvi_medio_estiagem_classe", "ndvi_tendencia_classe",
        "ndvi_delta_enso_classe", "ndvi_p90_classe",
        "classe_impermeabilizacao", "classe_app",
        "classe_distancia_urbana", "classe_ipvs",
        "indice_resiliencia_hidrologica", "indice_uso_solo_socioeconomico",
        "indice_prioridade_recuperacao",
    ]
    st.dataframe(
        df_final[colunas_exibir].sort_values(
            "indice_prioridade_recuperacao", ascending=False
        ).style.format({
            "indice_resiliencia_hidrologica": "{:.2f}",
            "indice_uso_solo_socioeconomico": "{:.2f}",
            "indice_prioridade_recuperacao": "{:.2f}",
        }),
        use_container_width=True,
        height=450,
    )

    geojson = pl.exportar_geojson(df_final)
    import json
    st.download_button(
        "⬇️ Baixar resultado (GeoJSON)",
        data=json.dumps(geojson, ensure_ascii=False, indent=2),
        file_name="nascentes_prioridade.geojson",
        mime="application/geo+json",
    )

st.markdown("---")
with st.expander("ℹ️ Sobre a metodologia e as limitações deste protótipo"):
    st.markdown(
        """
**O que este protótipo calcula de verdade:**
- Delimitação da Zona de Umidade da Superfície (SMZ) por célula de 30 m via P90 do NDVI + limiar T;
- Indicadores do Bloco 1 (Vegetação): NDVI médio de estiagem, tendência temporal (β), amplitude ENSO (ΔNDVI), NDVIP90;
- Indicadores do Bloco 2 (Uso do Solo/Socioeconômico): taxa de impermeabilização, adequação da APP, distância à mancha urbana, IPVS;
- Síntese do Bloco 1 por **PCA** → Índice de Resiliência Hidrológica;
- Normalização (min-max) e ponderação direta do Bloco 2 → Índice de Uso do Solo/Socioeconômico;
- Integração por **AHP** (pesos ajustáveis) → Índice de Prioridade de Recuperação;
- Exportação para GeoJSON.

**O que foi substituído para o protótipo funcionar sem credenciais externas:**
- Nascentes, sub-bacias, séries de NDVI, precipitação, uso do solo e IPVS são **sintéticos** (ver `data_synthetic.py`), no lugar de DataGEO, Google Earth Engine, INMET, MapBiomas, GeoSampa e Fundação SEADE.
- O **Bloco 3 (Governança)** foi excluído por instrução explícita (sem acesso aos shapefiles do Programa Mananciais, PSA Mananciais e APP-OIDA 2026). O AHP roda apenas com os Blocos 1 e 2.

Para produção, basta substituir as funções de `data_synthetic.py` por conectores reais — o restante do pipeline (`pipeline.py`) já está pronto para consumir dados reais com a mesma estrutura.
        """
    )
