import math
from fastapi import FastAPI
import pandas as pd
import numpy as np

app = FastAPI(
    title="API de Tendências Musicais e Engajamento",
    description="Algoritmo avançado para detecção de hype de artistas e automação de estratégias."
)

@app.get("/")
def home():
    return {
        "status": "Online",
        "mensagem": "Algoritmo de Tendências Musicais rodando com todas as melhorias aplicadas."
    }

@app.get("/tendencias")
def obter_tendencias():
    # 1. Base de Dados Simulada (Dados brutos extraídos de APIs das redes)
    dados_musica = {
        'Artista': ['Anitta', 'Anitta', 'Alok', 'Caetano Veloso', 'Matuê', 'Alok', 'Matuê'],
        'Faixa_ou_Assunto': ['Novo Single', 'Show de Ontem', 'Remix Eletrônico', 'Turnê de Despedida', 'Álbum Novo', 'Podcast Track', 'Prévia no TikTok'],
        'Mençoes': [1500, 800, 450, 2100, 5000, 200, 3500],
        'Compartilhamentos': [900, 150, 80, 1200, 4100, 30, 2800], 
        'Salves_Favoritos': [600, 90, 110, 850, 3200, 15, 1900],   
        'Horas_Desde_Lancamento': [3, 12, 5, 24, 2, 48, 1]          
    }
    
    df = pd.DataFrame(dados_musica)
    
    # 2. Pesos de Engajamento Otimizados
    PESO_MENCAO = 1.0
    PESO_COMPARTILHAR = 3.5  
    PESO_SALVAR = 4.0         
    
    df['Engajamento_Ponderado'] = (
        (df['Mençoes'] * PESO_MENCAO) + 
        (df['Compartilhamentos'] * PESO_COMPARTILHAR) + 
        (df['Salves_Favoritos'] * PESO_SALVAR)
    )
    
    # 3. Cálculo do Trend Score com Decaimento Temporal Exponencial (gama = 1.7)
    FATOR_DECAIMENTO = 1.7  
    df['Trend_Score'] = df['Engajamento_Ponderado'] / (df['Horas_Desde_Lancamento'] + 1) ** FATOR_DECAIMENTO
    
    # 4. Agrupamento e Consolidação por Artista
    ranking = df.groupby('Artista').agg(
        Volume_Total_Mençoes=('Mençoes', 'sum'),
        Total_Salves=('Salves_Favoritos', 'sum'),
        Hype_Score=('Trend_Score', 'sum'),
        Idade_Media_Horas=('Horas_Desde_Lancamento', 'mean')
    ).sort_values(by='Hype_Score', ascending=False).reset_index()
    
    # 5. Correção de Truncamento (Garante que limites numéricos não falhem por arredondamento)
    def truncar_score(valor, casas_decimais=2):
        fator = 10 ** casas_decimais
        return math.floor(valor * fator) / fator

    ranking['Hype_Score'] = ranking['Hype_Score'].apply(lambda x: truncar_score(x, 2))
    
    # 6. Extração dos Líderes para Lógica de Competição (Guerra de Fandoms)
    top_2_artistas = ranking['Artista'].head(2).tolist()
    
    # 7. Motor de Sugestão de Ações Automatizadas
    def gerar_estrategia_engajamento(row):
        acoes = []
        
        # Ação: Ultra Hype / Áudios em Alta
        if row['Hype_Score'] > 5000 and row['Idade_Media_Horas'] <= 6:
            acoes.append("🔥 ULTRA HYPE: Crie Reels/Shorts com o áudio oficial AGORA. O algoritmo das redes está entregando organicamente.")
        
        # Ação: Guerra de Fandoms
        if row['Artista'] in top_2_artistas and row['Volume_Total_Mençoes'] > 2000:
            acoes.append("⚔️ GUERRA DE FANDOMS: Crie uma enquete disputada no Instagram/X comparando este artista com o outro líder do ranking para gerar comentários em massa.")
            
        # Ação: Gatilho de Salvamento
        if row['Total_Salves'] > 1500:
            acoes.append("💾 CTA DE SALVAMENTO: Poste um carrossel informativo (ex: tracklist, datas) e termine com 'Salve este post para não esquecer!'.")
            
        # Ação: Efeito Nostalgia / Resgate de Catálogo
        if row['Hype_Score'] > 500 and row['Idade_Media_Horas'] >= 24:
            acoes.append("⏳ NOSTALGIA/TBT: O assunto continua relevante mesmo após 24h. Vale um post contextualizando a história ou conquistas do artista.")
            
        if not acoes:
            acoes.append("📊 MONITORAMENTO: Mantenha postagens padrão em tweets rápidos.")
            
        return " | ".join(acoes)
    
    ranking['Ação_Sugerida'] = ranking.apply(gerar_estrategia_engajamento, axis=1)
    
    # Retorna o resultado limpo e estruturado em formato JSON para a web
    return ranking.to_dict(orient="records")