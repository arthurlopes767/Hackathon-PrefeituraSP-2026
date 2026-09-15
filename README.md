# Protótipo — Priorização da Recuperação de Nascentes Urbanas (SP)

Protótipo em **Streamlit** do sistema de apoio à decisão baseado em análise
multicritério descrito no projeto do Hackathon Climático 2026: Água.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abra o endereço indicado no terminal (geralmente `http://localhost:8501`).

## Estrutura

| Arquivo | Camada da arquitetura | O que faz |
|---|---|---|
| `config.yaml` | Configuração | Limiar T da SMZ, limite de estiagem, anos ENSO, pesos default do AHP, faixas de classificação. |
| `data_synthetic.py` | Coleta de dados (**substituta sintética**) | Gera nascentes, séries de NDVI/precipitação e atributos de uso do solo/IPVS sintéticos. |
| `pipeline.py` | Pré-processamento, indicadores, modelagem, armazenamento | Delimita a SMZ (P90 NDVI + limiar T), calcula os Blocos 1 e 2, roda PCA (Bloco 1) e prepara o AHP; exporta GeoJSON. |
| `app.py` | Visualização | Dashboard Streamlit com dois mapas Folium (resiliência e prioridade), sliders de peso do AHP e tabela de indicadores. |

## Decisões importantes tomadas neste protótipo

1. **Dados sintéticos.** Este ambiente não tem acesso a credenciais/fontes
   reais (DataGEO, Google Earth Engine, INMET, MapBiomas, GeoSampa, SEADE).
   `data_synthetic.py` gera um dataset com a mesma estrutura estatística que
   os dados reais teriam (séries mensais de NDVI 2014–2025 com sazonalidade,
   tendência e resposta a El Niño/La Niña; precipitação; uso do solo; IPVS),
   para que o restante do pipeline seja 100% funcional. **Para produção,
   basta reescrever as funções desse arquivo com os conectores reais** — as
   assinaturas de saída (colunas) já são as esperadas por `pipeline.py`.

2. **Bloco 3 (Governança) desativado.** Por instrução explícita, o
   indicador de sobreposição com Programa Mananciais / PSA Mananciais /
   APP-OIDA 2026 **não foi implementado**. O AHP final combina apenas o
   Bloco 1 (Resiliência Hidrológica) e o Bloco 2 (Uso do Solo e
   Socioeconômico), com pesos normalizados entre os dois. Isso está
   documentado em `config.yaml` (`governanca.incluir_bloco_3: false`) e no
   próprio dashboard.

3. **Classificação de "Distância de áreas urbanizadas".** O material do
   projeto pedia uma pesquisa sobre a faixa de classificação (Baixa/Média/
   Alta) apropriada para esse indicador especificamente. Não há uma
   convenção própria publicada pelas fontes citadas (GeoSampa/MapBiomas)
   para esse corte; por consistência com os demais indicadores normalizados
   do Bloco 2 — que seguem todos a mesma regra 0–0,33 / >0,33–0,66 /
   >0,66–1,00 após normalização min-max — adotou-se a mesma faixa aqui.
   Ajustável em `config.yaml` se a equipe técnica definir um critério
   próprio.

4. **Grade de células por nascente.** Como não há um raster real de 30 m
   ao redor de cada nascente neste ambiente, cada nascente sintética recebe
   uma grade 5×5 de "células" com séries de NDVI levemente distintas entre
   si (mesma lógica de heterogeneidade intra-SMZ), sobre a qual o P90 e o
   limiar T são aplicados célula a célula, exatamente como descrito na
   metodologia (Cartwright & Johnson, 2018, adaptado).

5. **GeoJSON sem GeoPandas.** Para manter o protótipo leve e sem
   dependências geoespaciais pesadas (compatível com o escopo do
   hackathon), o GeoJSON é montado diretamente com Python padrão
   (`pipeline.exportar_geojson`), no formato `FeatureCollection` com
   geometrias `Point`. Pode ser baixado direto pelo botão no dashboard.

## Próximos passos para ir a produção

- Substituir `data_synthetic.py` por:
  - leitura do shapefile de nascentes do DataGEO (geopandas);
  - malha real de sub-bacias hidrográficas para a clusterização;
  - extração de NDVI via `earthengine-api` (Landsat 7/8/9, 30 m);
  - séries de precipitação do INMET (API ou download);
  - MapBiomas + GeoSampa para uso do solo e APP;
  - IPVS da Fundação SEADE por setor censitário.
- Reativar o Bloco 3 (Governança) assim que os shapefiles do Programa
  Mananciais, PSA Mananciais e APP-OIDA 2026 estiverem disponíveis, e
  redistribuir os pesos do AHP entre os três blocos.
- Avaliar a migração da persistência de GeoJSON único para um banco
  geoespacial (ex. PostGIS), se o volume de nascentes crescer além do
  escopo do protótipo.
