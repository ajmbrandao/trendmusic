"""
API de Tendências Musicais e Engajamento — DADOS REAIS
------------------------------------------------------
Fontes de dados reais:
  * YouTube Data API v3 -> views, likes, comentários e recência por vídeo
  * Last.fm API         -> ouvintes e reproduções totais por artista (popularidade)

O Hype Score combina engajamento RECENTE (YouTube, com decaimento temporal)
modulado pela popularidade GLOBAL do artista (Last.fm).

Requer duas variáveis de ambiente (definir no painel do Render -> Environment):
  YOUTUBE_API_KEY -> Google Cloud Console (ativar "YouTube Data API v3")
  LASTFM_API_KEY  -> https://www.last.fm/api/account/create
"""

import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("tendencias")

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

ARTISTAS = ["CATWEN", "Dua Lipa", "Zara Larsson"]

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")

YOUTUBE_BASE = "https://www.googleapis.com/youtube/v3"
LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"

MAX_VIDEOS_POR_ARTISTA = 3
HTTP_TIMEOUT = 10  # segundos

# Pesos de engajamento (YouTube). Likes e comentários valem mais que views
# porque exigem mais intenção do utilizador.
PESO_VIEW = 1.0
PESO_LIKE = 5.0
PESO_COMENTARIO = 10.0

# Decaimento temporal exponencial aplicado ao Trend Score
FATOR_DECAIMENTO = 1.7

# Peso da popularidade da Last.fm na modulação do Hype Score (escala log, para
# que números enormes de ouvintes não esmaguem por completo o sinal recente).
PESO_LASTFM_POP = 0.15

# Limiares das regras de estratégia.
# IMPORTANTE: com dados reais as ordens de grandeza mudam muito (views podem ir
# aos milhões). Estes valores são PONTOS DE PARTIDA — calibra-os depois de veres
# os números reais que o /tendencias devolve.
LIMIAR_FRESCO_HORAS = 72            # "fresco" = média de idade <= 3 dias
LIMIAR_NOSTALGIA_IDADE = 24 * 30    # conteúdo já com >= ~30 dias
LIMIAR_NOSTALGIA_OUVINTES = 500_000  # artista globalmente popular (Last.fm)
LIMIAR_CTA_LIKES = 50_000           # total de likes para CTA de salvamento

# Cache em memória: a search.list custa 100 unidades de quota; sem cache
# esgotarias as 10.000/dia em poucas dezenas de pedidos.
CACHE_TTL_SEGUNDOS = 15 * 60
_cache = {"timestamp": 0.0, "dados": None}

app = FastAPI(
    title="API de Tendências Musicais e Engajamento",
    description="Hype de artistas a partir de dados reais do YouTube e da Last.fm.",
)


# ---------------------------------------------------------------------------
# Modelo de resposta (documenta o contrato e força tipos nativos no JSON)
# ---------------------------------------------------------------------------

class TendenciaArtista(BaseModel):
    Artista: str
    Total_Views: int
    Total_Likes: int
    Total_Comentarios: int
    LastFm_Ouvintes: int
    LastFm_Plays: int
    Hype_Score: float
    Idade_Media_Horas: float
    Acao_Sugerida: str


# ---------------------------------------------------------------------------
# Acesso às APIs externas (isolado para ser fácil de testar/substituir)
# ---------------------------------------------------------------------------

def _http_get_json(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _horas_desde(iso_timestamp: str) -> float:
    """Converte um timestamp ISO 8601 (ex.: '2026-05-30T12:00:00Z') em horas decorridas."""
    publicado = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    delta = datetime.now(timezone.utc) - publicado
    return max(delta.total_seconds() / 3600.0, 0.0)


def buscar_videos_youtube(artista: str) -> List[dict]:
    """Procura os vídeos mais recentes do artista e devolve estatísticas reais."""
    # 1) search.list -> obter IDs de vídeos (custa 100 unidades de quota)
    busca = _http_get_json(
        f"{YOUTUBE_BASE}/search",
        {
            "key": YOUTUBE_API_KEY,
            "q": artista,
            "part": "id",
            "type": "video",
            "order": "date",
            "maxResults": MAX_VIDEOS_POR_ARTISTA,
        },
    )
    ids = [
        item["id"]["videoId"]
        for item in busca.get("items", [])
        if item.get("id", {}).get("videoId")
    ]
    if not ids:
        return []

    # 2) videos.list -> estatísticas reais (custa apenas 1 unidade)
    detalhes = _http_get_json(
        f"{YOUTUBE_BASE}/videos",
        {"key": YOUTUBE_API_KEY, "id": ",".join(ids), "part": "snippet,statistics"},
    )

    linhas = []
    for item in detalhes.get("items", []):
        stats = item.get("statistics", {})
        snippet = item.get("snippet", {})
        linhas.append(
            {
                "Artista": artista,
                "Faixa_ou_Assunto": snippet.get("title", "(sem título)"),
                "Views": int(stats.get("viewCount", 0)),
                # likeCount pode vir ausente se o criador esconder os gostos
                "Likes": int(stats.get("likeCount", 0)),
                "Comentarios": int(stats.get("commentCount", 0)),
                "Horas_Desde_Publicacao": _horas_desde(snippet["publishedAt"]),
            }
        )
    return linhas


def buscar_popularidade_lastfm(artista: str) -> dict:
    """Devolve ouvintes e reproduções totais do artista (Last.fm)."""
    dados = _http_get_json(
        LASTFM_BASE,
        {
            "method": "artist.getInfo",
            "artist": artista,
            "api_key": LASTFM_API_KEY,
            "format": "json",
            "autocorrect": 1,
        },
    )
    stats = dados.get("artist", {}).get("stats", {})
    return {
        "LastFm_Ouvintes": int(stats.get("listeners", 0)),
        "LastFm_Plays": int(stats.get("playcount", 0)),
    }


# ---------------------------------------------------------------------------
# Lógica de domínio (testável sem rede)
# ---------------------------------------------------------------------------

def truncar_score(valor: float, casas_decimais: int = 2) -> float:
    fator = 10 ** casas_decimais
    return math.floor(valor * fator) / fator


def calcular_ranking(videos: List[dict], popularidade: dict) -> pd.DataFrame:
    """Combina engajamento recente (YouTube) com popularidade total (Last.fm)."""
    if not videos:
        return pd.DataFrame()

    df = pd.DataFrame(videos)

    df["Engajamento_Ponderado"] = (
        df["Views"] * PESO_VIEW
        + df["Likes"] * PESO_LIKE
        + df["Comentarios"] * PESO_COMENTARIO
    )
    horas = df["Horas_Desde_Publicacao"].clip(lower=0)
    df["Trend_Score"] = df["Engajamento_Ponderado"] / (horas + 1) ** FATOR_DECAIMENTO

    ranking = (
        df.groupby("Artista")
        .agg(
            Total_Views=("Views", "sum"),
            Total_Likes=("Likes", "sum"),
            Total_Comentarios=("Comentarios", "sum"),
            Trend_Score_Base=("Trend_Score", "sum"),
            Idade_Media_Horas=("Horas_Desde_Publicacao", "mean"),
        )
        .reset_index()
    )

    # Anexar popularidade da Last.fm
    ranking["LastFm_Ouvintes"] = ranking["Artista"].map(
        lambda a: popularidade.get(a, {}).get("LastFm_Ouvintes", 0)
    )
    ranking["LastFm_Plays"] = ranking["Artista"].map(
        lambda a: popularidade.get(a, {}).get("LastFm_Plays", 0)
    )

    # Modular o sinal recente do YouTube pela popularidade global da Last.fm
    fator_pop = 1 + PESO_LASTFM_POP * np.log10(ranking["LastFm_Ouvintes"].clip(lower=0) + 1)
    ranking["Hype_Score"] = (ranking["Trend_Score_Base"] * fator_pop).apply(
        lambda x: truncar_score(x, 2)
    )

    ranking = ranking.sort_values(by="Hype_Score", ascending=False).reset_index(drop=True)
    ranking["Idade_Media_Horas"] = ranking["Idade_Media_Horas"].round(1)
    return ranking


def gerar_estrategias(ranking: pd.DataFrame) -> pd.DataFrame:
    """Anexa a coluna 'Acao_Sugerida' usando posição no ranking + sinais das duas fontes."""
    if ranking.empty:
        return ranking
    ranking = ranking.copy()

    def estrategia(posicao: int, row) -> str:
        acoes = []

        # Ultra Hype: lidera o ranking e tem conteúdo fresco
        if posicao == 0 and row["Idade_Media_Horas"] <= LIMIAR_FRESCO_HORAS:
            acoes.append(
                "🔥 ULTRA HYPE: Crie Reels/Shorts com o áudio oficial AGORA. "
                "O algoritmo das redes está a entregar organicamente."
            )

        # Guerra de fandoms: entre os dois líderes do ranking
        if posicao <= 1:
            acoes.append(
                "⚔️ GUERRA DE FANDOMS: Crie uma enquete disputada no Instagram/X "
                "comparando este artista com o outro líder do ranking."
            )

        # CTA de salvamento: muito engajamento de likes (proxy de 'salvar')
        if row["Total_Likes"] >= LIMIAR_CTA_LIKES:
            acoes.append(
                "💾 CTA DE SALVAMENTO: Poste um carrossel informativo e termine com "
                "'Salve este post para não esquecer!'."
            )

        # Nostalgia/TBT: conteúdo recente já antigo MAS artista globalmente popular
        if (
            row["Idade_Media_Horas"] >= LIMIAR_NOSTALGIA_IDADE
            and row["LastFm_Ouvintes"] >= LIMIAR_NOSTALGIA_OUVINTES
        ):
            acoes.append(
                "⏳ NOSTALGIA/TBT: Sem lançamento recente, mas o artista mantém base "
                "enorme. Vale um post de catálogo / efeméride."
            )

        if not acoes:
            acoes.append("📊 MONITORAMENTO: Mantenha postagens padrão em tweets rápidos.")

        return " | ".join(acoes)

    ranking["Acao_Sugerida"] = [estrategia(i, row) for i, row in ranking.iterrows()]
    return ranking


def para_tipos_nativos(ranking: pd.DataFrame) -> List[dict]:
    """Converte escalares NumPy em tipos nativos do Python (serialização portável)."""
    registos = ranking.to_dict(orient="records")
    return [
        {k: (v.item() if isinstance(v, np.generic) else v) for k, v in registo.items()}
        for registo in registos
    ]


def coletar_dados_reais() -> List[dict]:
    if not YOUTUBE_API_KEY or not LASTFM_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Faltam YOUTUBE_API_KEY e/ou LASTFM_API_KEY nas variáveis de ambiente.",
        )

    videos: List[dict] = []
    popularidade: dict = {}
    for artista in ARTISTAS:
        try:
            videos.extend(buscar_videos_youtube(artista))
        except Exception as exc:  # uma API falhar não deve derrubar o pedido todo
            logger.warning("YouTube falhou para %s: %s", artista, exc)
        try:
            popularidade[artista] = buscar_popularidade_lastfm(artista)
        except Exception as exc:
            logger.warning("Last.fm falhou para %s: %s", artista, exc)

    if not videos:
        raise HTTPException(
            status_code=502,
            detail="Não foi possível obter dados do YouTube para nenhum artista.",
        )

    ranking = calcular_ranking(videos, popularidade)
    ranking = gerar_estrategias(ranking)
    return para_tipos_nativos(ranking)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {
        "status": "Online",
        "mensagem": "Algoritmo de Tendências Musicais com dados reais (YouTube + Last.fm).",
    }


@app.get("/tendencias", response_model=List[TendenciaArtista])
def obter_tendencias(forcar_atualizacao: bool = False):
    agora = time.time()
    cache_valido = (
        _cache["dados"] is not None
        and (agora - _cache["timestamp"]) < CACHE_TTL_SEGUNDOS
    )
    if not forcar_atualizacao and cache_valido:
        return _cache["dados"]

    dados = coletar_dados_reais()
    _cache["dados"] = dados
    _cache["timestamp"] = agora
    return dados
