"""
API de Tendências Musicais e Engajamento
-----------------------------------------
Algoritmo para deteção de hype de artistas e sugestão de estratégias.

Versão revista:
  - Correção da regra "Nostalgia/TBT" (antes era código morto).
  - Robustez na divisão (impede horas negativas -> inf/NaN no JSON).
  - Serialização portável entre versões do pandas (cast de escalares NumPy).
  - Limiares centralizados em constantes nomeadas.
  - Nomes de campos ASCII e estáveis no contrato da API.
  - Lógica separada em funções testáveis isoladamente.
"""

import math
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Parâmetros do modelo (centralizados para facilitar a afinação)
# ---------------------------------------------------------------------------

# Pesos de engajamento
PESO_MENCAO = 1.0
PESO_COMPARTILHAR = 3.5
PESO_SALVAR = 4.0

# Decaimento temporal exponencial aplicado ao Trend Score
FATOR_DECAIMENTO = 1.7

# Limiares das regras de estratégia (ajustar conforme o comportamento real)
LIMIAR_ULTRA_HYPE_SCORE = 5000      # Hype mínimo para "Ultra Hype"
LIMIAR_ULTRA_HYPE_IDADE = 6         # Idade máxima (h) para conteúdo "fresco"
LIMIAR_GUERRA_MENCOES = 2000        # Menções mínimas para disputa de fandoms
LIMIAR_SALVAR = 1500                # Salves mínimos para CTA de salvamento
LIMIAR_NOSTALGIA_IDADE = 24         # Idade mínima (h) para considerar "TBT"
LIMIAR_NOSTALGIA_MENCOES = 1500     # Engajamento absoluto mínimo p/ nostalgia

app = FastAPI(
    title="API de Tendências Musicais e Engajamento",
    description="Algoritmo avançado para deteção de hype de artistas e automação de estratégias.",
)


# ---------------------------------------------------------------------------
# Modelo de resposta (documenta o contrato e força tipos nativos no JSON)
# ---------------------------------------------------------------------------

class TendenciaArtista(BaseModel):
    Artista: str
    Volume_Total_Mencoes: int
    Total_Salves: int
    Hype_Score: float
    Idade_Media_Horas: float
    Acao_Sugerida: str


# ---------------------------------------------------------------------------
# Funções de domínio (separadas para serem testáveis isoladamente)
# ---------------------------------------------------------------------------

def carregar_dados() -> pd.DataFrame:
    """Fonte de dados (simulada). Substituir pela ingestão real das APIs das redes."""
    dados_musica = {
        "Artista": [
            "CATWEN", "CATWEN", "Dua Lipa", "Zara Larsson",
        ],
        "Faixa_ou_Assunto": [
            "Novo Single", "Show de Ontem", "Remix Eletrônico", "Turnê de Despedida",
            "Álbum Novo", "Podcast Track", "Prévia no TikTok",
        ],
        "Mencoes": [1500, 800, 450, 2100, 5000, 200, 3500],
        "Compartilhamentos": [900, 150, 80, 1200, 4100, 30, 2800],
        "Salves_Favoritos": [600, 90, 110, 850, 3200, 15, 1900],
        "Horas_Desde_Lancamento": [3, 12, 5, 24, 2, 48, 1],
    }
    return pd.DataFrame(dados_musica)


def truncar_score(valor: float, casas_decimais: int = 2) -> float:
    """Trunca (não arredonda) para evitar que limites numéricos falhem por arredondamento."""
    fator = 10 ** casas_decimais
    return math.floor(valor * fator) / fator


def calcular_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula engajamento ponderado, trend score com decaimento e consolida por artista."""
    df = df.copy()

    df["Engajamento_Ponderado"] = (
        df["Mencoes"] * PESO_MENCAO
        + df["Compartilhamentos"] * PESO_COMPARTILHAR
        + df["Salves_Favoritos"] * PESO_SALVAR
    )

    # Robustez: nunca permitir horas negativas (dados das APIs podem vir
    # dessincronizados / com fuso errado). Sem isto, uma hora == -1 daria
    # divisão por zero -> inf -> JSON inválido.
    horas = df["Horas_Desde_Lancamento"].clip(lower=0)
    df["Trend_Score"] = df["Engajamento_Ponderado"] / (horas + 1) ** FATOR_DECAIMENTO

    # Nota de design: somamos o Trend_Score por artista (mede presença total
    # no momento). Para medir apenas "a faixa mais quente agora", trocar
    # 'sum' por 'max' na agregação de Hype_Score.
    ranking = (
        df.groupby("Artista")
        .agg(
            Volume_Total_Mencoes=("Mencoes", "sum"),
            Total_Salves=("Salves_Favoritos", "sum"),
            Hype_Score=("Trend_Score", "sum"),
            Idade_Media_Horas=("Horas_Desde_Lancamento", "mean"),
        )
        .sort_values(by="Hype_Score", ascending=False)
        .reset_index()
    )

    ranking["Hype_Score"] = ranking["Hype_Score"].apply(lambda x: truncar_score(x, 2))
    return ranking


def gerar_estrategias(ranking: pd.DataFrame) -> pd.DataFrame:
    """Anexa a coluna 'Acao_Sugerida' com base nas regras de negócio."""
    ranking = ranking.copy()
    top_2_artistas = ranking["Artista"].head(2).tolist()

    def gerar_estrategia_engajamento(row) -> str:
        acoes = []

        # Ultra Hype: score muito alto em conteúdo fresco
        if (
            row["Hype_Score"] > LIMIAR_ULTRA_HYPE_SCORE
            and row["Idade_Media_Horas"] <= LIMIAR_ULTRA_HYPE_IDADE
        ):
            acoes.append(
                "🔥 ULTRA HYPE: Crie Reels/Shorts com o áudio oficial AGORA. "
                "O algoritmo das redes está a entregar organicamente."
            )

        # Guerra de fandoms: entre os líderes e com volume relevante
        if (
            row["Artista"] in top_2_artistas
            and row["Volume_Total_Mencoes"] > LIMIAR_GUERRA_MENCOES
        ):
            acoes.append(
                "⚔️ GUERRA DE FANDOMS: Crie uma enquete disputada no Instagram/X "
                "comparando este artista com o outro líder do ranking para gerar "
                "comentários em massa."
            )

        # CTA de salvamento
        if row["Total_Salves"] > LIMIAR_SALVAR:
            acoes.append(
                "💾 CTA DE SALVAMENTO: Poste um carrossel informativo (ex.: tracklist, "
                "datas) e termine com 'Salve este post para não esquecer!'."
            )

        # Nostalgia / resgate de catálogo: conteúdo já ANTIGO mas que ainda tem
        # engajamento ABSOLUTO elevado. Usamos o volume de menções (não decaído),
        # porque o Trend_Score decai demasiado para detetar relevância duradoura
        # — era por isso que a regra antiga (baseada em Hype_Score) nunca disparava.
        if (
            row["Idade_Media_Horas"] >= LIMIAR_NOSTALGIA_IDADE
            and row["Volume_Total_Mencoes"] >= LIMIAR_NOSTALGIA_MENCOES
        ):
            acoes.append(
                "⏳ NOSTALGIA/TBT: O assunto continua relevante mesmo após 24h. "
                "Vale um post contextualizando a história ou conquistas do artista."
            )

        if not acoes:
            acoes.append("📊 MONITORAMENTO: Mantenha postagens padrão em tweets rápidos.")

        return " | ".join(acoes)

    ranking["Acao_Sugerida"] = ranking.apply(gerar_estrategia_engajamento, axis=1)
    return ranking


def para_tipos_nativos(ranking: pd.DataFrame) -> List[dict]:
    """Converte escalares NumPy (np.int64/np.float64) em tipos nativos do Python.

    Em versões antigas do pandas, to_dict devolve escalares NumPy, que o encoder
    por omissão do FastAPI não consegue serializar (resulta em erro 500). Este
    cast torna a resposta portável independentemente da versão do pandas.
    """
    registos = ranking.to_dict(orient="records")
    return [
        {k: (v.item() if isinstance(v, np.generic) else v) for k, v in registo.items()}
        for registo in registos
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {
        "status": "Online",
        "mensagem": "Algoritmo de Tendências Musicais a correr com todas as melhorias aplicadas.",
    }


@app.get("/tendencias", response_model=List[TendenciaArtista])
def obter_tendencias():
    df = carregar_dados()
    ranking = calcular_ranking(df)
    ranking = gerar_estrategias(ranking)
    return para_tipos_nativos(ranking)
