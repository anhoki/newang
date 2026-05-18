# montana_rusa_selector.py
# Dashboard para simular la selección de candidatos + timing de entrada al open

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime, timedelta

st.set_page_config(page_title="Selector de Montaña Rusa + Timing Open", layout="wide")

# ============================================================
# 1. FUNCIONES DE AJUSTE EXPONENCIAL Y MÉTRICAS
# ============================================================

def exponencial_crecimiento(t, A, k, B):
    return A * np.exp(k * t) + B

def ema(series, alpha=0.3):
    result = np.zeros_like(series, dtype=float)
    if len(series) == 0:
        return result
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
    return result

def ajustar_exponencial(y_vals):
    x_vals = np.arange(len(y_vals))
    y_pos = np.maximum(y_vals, 0.01)
    
    y_max = y_pos.max()
    y_min = y_pos.min()
    A0 = max((y_max - y_min) * 0.5, 0.1)
    k0 = 0.1
    B0 = max(y_min, 0)
    
    try:
        params, _ = curve_fit(exponencial_crecimiento, x_vals, y_pos, 
                              p0=[A0, k0, B0], maxfev=5000)
        A, k, B = params
        
        y_pred = exponencial_crecimiento(x_vals, A, k, B)
        ss_res = np.sum((y_pos - y_pred) ** 2)
        ss_tot = np.sum((y_pos - np.mean(y_pos)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r2 = max(0, min(1, r2))
        
        # Longitud de arco
        horizonte = len(y_vals) - 1
        pasos = 100
        dt = horizonte / pasos if horizonte > 0 else 1
        longitud = 0
        for i in range(pasos):
            t = i * dt
            t_next = (i + 1) * dt
            ft = exponencial_crecimiento(t, A, k, B)
            ft_next = exponencial_crecimiento(t_next, A, k, B)
            dy = ft_next - ft
            dx = dt
            longitud += np.sqrt(dx*dx + dy*dy)
        
        # Velocidad inicial y final
        velocidad_inicial = A * k
        velocidad_final = A * k * np.exp(k * (len(y_vals)-1)) if k != 0 else 0
        aceleracion_promedio = (velocidad_final - velocidad_inicial) / (len(y_vals)-1) if len(y_vals) > 1 else 0
        
        # Ímpetu (explosividad)
        impetu = abs(A * k * np.exp(k * (len(y_vals)-1))) * longitud
        
        return {
            'k': k, 'A': A, 'B': B, 'r2': r2,
            'longitud': longitud,
            'vel_inicial': velocidad_inicial,
            'vel_final': velocidad_final,
            'aceleracion': aceleracion_promedio,
            'impetu': impetu,
            'y_pred': y_pred
        }
    except Exception as e:
        return {
            'k': 0, 'A': 0, 'B': 0, 'r2': 0,
            'longitud': 0, 'vel_inicial': 0, 'vel_final': 0,
            'aceleracion': 0, 'impetu': 0, 'y_pred': np.zeros_like(y_vals)
        }

# ============================================================
# 2. GENERAR DATOS DE EJEMPLO (DÍA ANTERIOR + SIMULACIÓN OPEN)
# ============================================================

def generar_datos_ejemplo():
    """Genera datos simulados para 5 activos con diferentes comportamientos"""
    
    np.random.seed(42)
    
    activos = ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOGL']
    velas = list(range(20))
    
    datos = []
    
    # Comportamientos específicos
    # TSLA: explosivo al final (k grande)
    # AAPL: crecimiento constante
    # MSFT: decaimiento
    # AMZN: crecimiento suave
    # GOOGL: errático
    
    for i, ticker in enumerate(activos):
        if ticker == 'TSLA':
            votos = [2,2,3,3,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5]
            precios = [250 + j*1.5 for j in range(20)]
        elif ticker == 'AAPL':
            votos = [1,1,2,2,3,3,3,4,4,4,4,5,5,5,4,4,4,5,5,5]
            precios = [179 + j*0.3 for j in range(20)]
        elif ticker == 'MSFT':
            votos = [5,5,4,4,3,3,3,2,2,2,2,1,1,1,2,2,2,1,1,1]
            precios = [420 - j*0.5 for j in range(20)]
        elif ticker == 'AMZN':
            votos = [2,2,3,3,3,4,4,4,4,4,4,4,4,4,3,3,4,4,4,4]
            precios = [178 + j*0.2 for j in range(20)]
        else:  # GOOGL
            votos = [1,3,1,3,5,3,1,3,5,3,1,3,5,3,1,3,5,3,1,3]
            precios = [165 + np.sin(j/3)*2 for j in range(20)]
        
        for vela, voto, precio in zip(velas, votos, precios):
            datos.append({
                'Ticker': ticker,
                'Vela': vela,
                'Votos_Neto': voto,
                'Precio': precio,
                'Volumen': np.random.uniform(2, 8)
            })
    
    return pd.DataFrame(datos)

def simular_open(df, ticker, sesgo):
    """Simula las primeras 2 velas del open para un ticker dado"""
    
    sub = df[df['Ticker'] == ticker]
    precio_cierre_ayer = sub['Precio'].iloc[-1]
    votos_cierre_ayer = sub['Votos_Neto'].iloc[-1]
    volumen_promedio = sub['Volumen'].mean()
    
    # Simular gap según el sesgo y el comportamiento histórico
    if sesgo == 'BULLISH':
        gap_pct = np.random.uniform(0.001, 0.01)
        direccion = 1
    else:
        gap_pct = np.random.uniform(-0.01, -0.001)
        direccion = -1
    
    # Velocidad y aceleración simuladas
    precio_open = precio_cierre_ayer * (1 + gap_pct)
    
    # Primera vela (5 min)
    velocidad_1 = np.random.uniform(0.001, 0.005) * direccion
    precio_vela1 = precio_open * (1 + velocidad_1)
    volumen_vela1 = volumen_promedio * np.random.uniform(1.2, 2.5)
    
    # Segunda vela (5 min)
    aceleracion = np.random.uniform(-0.002, 0.008) * direccion
    velocidad_2 = velocidad_1 + aceleracion
    precio_vela2 = precio_vela1 * (1 + velocidad_2)
    volumen_vela2 = volumen_promedio * np.random.uniform(1.0, 2.0)
    
    return {
        'ticker': ticker,
        'precio_cierre_ayer': precio_cierre_ayer,
        'votos_cierre_ayer': votos_cierre_ayer,
        'gap_pct': gap_pct,
        'precio_open': precio_open,
        'velocidad_vela1': velocidad_1,
        'precio_vela1': precio_vela1,
        'volumen_vela1': volumen_vela1,
        'velocidad_vela2': velocidad_2,
        'precio_vela2': precio_vela2,
        'volumen_vela2': volumen_vela2,
        'aceleracion': aceleracion,
        'volumen_relativo_1': volumen_vela1 / volumen_promedio,
        'volumen_relativo_2': volumen_vela2 / volumen_promedio,
    }

# ============================================================
# 3. CARGA DE DATOS
# ============================================================

st.title("🎢 Selector de Montaña Rusa + Timing Open")
st.markdown("**Fase A:** Selección de candidatos con datos del día anterior")
st.markdown("**Fase B:** Simulación de entrada en las primeras 2 velas del open")

opcion = st.radio("Origen de datos", ["Usar datos de ejemplo", "Subir CSV con mis datos"])

df = None

if opcion == "Subir CSV con mis datos":
    archivo = st.file_uploader("Sube un CSV con columnas: Ticker, Vela, Votos_Neto, Precio, Volumen", type="csv")
    if archivo:
        df = pd.read_csv(archivo)
        st.success(f"✅ Datos cargados: {df['Ticker'].nunique()} activos, {len(df)} filas")
    else:
        st.warning("Esperando archivo... usando datos de ejemplo")
        opcion = "Usar datos de ejemplo"

if opcion == "Usar datos de ejemplo":
    df = generar_datos_ejemplo()
    st.info("📊 Usando datos de ejemplo simulados")

if df is None:
    st.stop()

# ============================================================
# 4. FASE A: SELECCIÓN DE CANDIDATOS (DÍA ANTERIOR)
# ============================================================

st.header("📊 FASE A: Selección de candidatos (datos día anterior)")

# Calcular EMA5
df['EMA5'] = df.groupby('Ticker')['Votos_Neto'].transform(lambda x: ema(x.values))

# Ajuste exponencial por activo
resultados = []
tickers = df['Ticker'].unique()

with st.spinner("Calculando curvas exponenciales..."):
    for ticker in tickers:
        sub = df[df['Ticker'] == ticker].copy()
        y_vals = sub['EMA5'].values
        
        ajuste = ajustar_exponencial(y_vals)
        
        resultados.append({
            "Ticker": ticker,
            "k": ajuste['k'],
            "A": ajuste['A'],
            "R²": ajuste['r2'],
            "Longitud_arco": ajuste['longitud'],
            "Velocidad_inicial": ajuste['vel_inicial'],
            "Velocidad_final": ajuste['vel_final'],
            "Aceleración": ajuste['aceleracion'],
            "Ímpetu": ajuste['impetu'],
            "y_pred": ajuste['y_pred'],
            "y_real": y_vals,
            "velas": sub['Vela'].values
        })

df_ranking = pd.DataFrame(resultados)

# Calcular puntaje para ranking
max_long = df_ranking['Longitud_arco'].max() if df_ranking['Longitud_arco'].max() > 0 else 1
max_k = df_ranking['k'].max() if df_ranking['k'].max() > 0 else 1
max_impetu = df_ranking['Ímpetu'].max() if df_ranking['Ímpetu'].max() > 0 else 1
max_vel_final = df_ranking['Velocidad_final'].max() if df_ranking['Velocidad_final'].max() > 0 else 1

# Puntaje combina: ímpetu (40%), velocidad final (30%), longitud (20%), k (10%)
df_ranking['Puntaje'] = (
    (df_ranking['Ímpetu'] / max_impetu) * 0.4 +
    (df_ranking['Velocidad_final'] / max_vel_final) * 0.3 +
    (df_ranking['Longitud_arco'] / max_long) * 0.2 +
    (df_ranking['k'] / max_k) * 0.1
)
df_ranking = df_ranking.sort_values('Puntaje', ascending=False)
df_ranking['Puntaje'] = df_ranking['Puntaje'] / df_ranking['Puntaje'].max()

# Mostrar ranking
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("🏆 Ranking de candidatos")
    
    display_df = df_ranking[['Ticker', 'k', 'R²', 'Longitud_arco', 'Velocidad_final', 'Ímpetu', 'Puntaje']].copy()
    display_df['k'] = display_df['k'].map(lambda x: f"{x:.3f}")
    display_df['R²'] = display_df['R²'].map(lambda x: f"{x:.2f}")
    display_df['Longitud_arco'] = display_df['Longitud_arco'].map(lambda x: f"{x:.2f}")
    display_df['Velocidad_final'] = display_df['Velocidad_final'].map(lambda x: f"{x:.3f}")
    display_df['Ímpetu'] = display_df['Ímpetu'].map(lambda x: f"{x:.2f}")
    display_df['Puntaje'] = display_df['Puntaje'].map(lambda x: f"{x:.0%}")
    
    st.dataframe(display_df, use_container_width=True)
    
    mejor = df_ranking.iloc[0]
    st.success(f"🎢 **MEJOR CANDIDATO (Fase A): {mejor['Ticker']}**")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Ímpetu", f"{mejor['Ímpetu']:.2f}")
    col_b.metric("Velocidad final", f"{mejor['Velocidad_final']:.3f}")
    col_c.metric("k (curvatura)", f"{mejor['k']:.3f}")

with col2:
    st.subheader("📈 Curvas exponenciales")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, row in df_ranking.iterrows():
        ticker = row['Ticker']
        sub = df[df['Ticker'] == ticker]
        color = colores[i % len(colores)]
        ax.plot(sub['Vela'], sub['EMA5'], 'o-', color=color, label=f"{ticker} (real)", alpha=0.7, markersize=4)
        ax.plot(sub['Vela'], row['y_pred'], '--', color=color, label=f"{ticker} (exp)", alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel("Vela (tiempo)")
    ax.set_ylabel("Votos Netos Suavizados (EMA5)")
    ax.set_title("Comparativa de curvas reales vs exponenciales")
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ============================================================
# 5. FASE B: SIMULACIÓN DE TIMING AL OPEN
# ============================================================

st.header("⏰ FASE B: Simulación de entrada al open")

st.markdown("Simula cómo se comportarían los candidatos en las primeras 2 velas de 5 minutos")

# Seleccionar candidatos para simular (top 3)
top_candidatos = df_ranking.head(3)['Ticker'].tolist()

# Parámetros de simulación
st.subheader("⚙️ Parámetros de la simulación")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    sesgo = st.selectbox("Sesgo del día", ["BULLISH (buscar CALL)", "BEARISH (buscar PUT)"])
with col_s2:
    velocidad_min_entrada = st.number_input("Velocidad mínima vela 1 (%/min)", value=0.15, step=0.05, format="%.2f") / 100
with col_s3:
    volumen_min_entrada = st.number_input("Volumen mínimo relativo", value=1.5, step=0.1, format="%.1f")

direccion = 1 if "BULLISH" in sesgo else -1

# Botón para simular
if st.button("🔄 Simular apertura de mercado", use_container_width=True):
    
    st.subheader("📊 Resultados de la simulación por candidato")
    
    resultados_open = []
    
    for ticker in top_candidatos:
        sim = simular_open(df, ticker, sesgo)
        resultados_open.append(sim)
    
    # Crear DataFrame con resultados
    df_open = pd.DataFrame(resultados_open)
    
    # Calcular puntaje de timing
    df_open['Puntaje_timing'] = (
        (df_open['gap_pct'] * direccion) * 0.3 +
        (df_open['velocidad_vela1'] * direccion) * 0.3 +
        (df_open['aceleracion'] * direccion) * 0.2 +
        (df_open['volumen_relativo_1'] / 3) * 0.2
    )
    # Normalizar entre 0 y 1
    min_punt = df_open['Puntaje_timing'].min()
    max_punt = df_open['Puntaje_timing'].max()
    if max_punt > min_punt:
        df_open['Puntaje_timing_norm'] = (df_open['Puntaje_timing'] - min_punt) / (max_punt - min_punt)
    else:
        df_open['Puntaje_timing_norm'] = 0.5
    
    # Mostrar resultados
    for i, row in df_open.iterrows():
        with st.expander(f"📈 {row['ticker']} - Gap: {row['gap_pct']*100:.2f}% | Velocidad: {row['velocidad_vela1']*100:.2f}%/min | Aceleración: {row['aceleracion']*100:.2f}%/min²"):
            
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Gap vs cierre", f"{row['gap_pct']*100:.2f}%", 
                         delta="favorable" if row['gap_pct']*direccion > 0 else "desfavorable")
                st.metric("Precio open", f"${row['precio_open']:.2f}")
            
            with col_b:
                st.metric("Velocidad vela 1", f"{row['velocidad_vela1']*100:.2f}%/min",
                         delta="✅ OK" if row['velocidad_vela1']*direccion > velocidad_min_entrada else "❌ baja")
                st.metric("Volumen vela 1", f"{row['volumen_relativo_1']:.1f}x promedio")
            
            with col_c:
                st.metric("Aceleración", f"{row['aceleracion']*100:.2f}%/min²",
                         delta="acelerando" if row['aceleracion']*direccion > 0 else "frenando")
                st.metric("Volumen vela 2", f"{row['volumen_relativo_2']:.1f}x promedio")
            
            with col_d:
                st.metric("Velocidad vela 2", f"{row['velocidad_vela2']*100:.2f}%/min")
                st.metric("Precio vela 2", f"${row['precio_vela2']:.2f}")
            
            # Señal de entrada
            st.markdown("---")
            
            condiciones = []
            if row['gap_pct'] * direccion > 0:
                condiciones.append("✅ Gap favorable")
            else:
                condiciones.append("❌ Gap en contra")
            
            if row['velocidad_vela1'] * direccion > velocidad_min_entrada:
                condiciones.append("✅ Velocidad suficiente")
            else:
                condiciones.append("❌ Velocidad insuficiente")
            
            if row['aceleracion'] * direccion > 0:
                condiciones.append("✅ Acelerando (gana fuerza)")
            else:
                condiciones.append("⚠️ Frenando (pierde fuerza)")
            
            if row['volumen_relativo_1'] > volumen_min_entrada:
                condiciones.append("✅ Volumen alto")
            else:
                condiciones.append("⚠️ Volumen bajo")
            
            st.write("**Condiciones de entrada:**")
            for cond in condiciones:
                st.write(cond)
            
            # Decisión
            if (row['gap_pct'] * direccion > 0 and 
                row['velocidad_vela1'] * direccion > velocidad_min_entrada and
                row['volumen_relativo_1'] > volumen_min_entrada):
                st.success(f"🎯 **DECISIÓN: ENTRAR en {row['ticker']} en la vela 2**")
                st.caption("Todas las condiciones clave se cumplen")
            elif (row['gap_pct'] * direccion > 0 and 
                  row['velocidad_vela1'] * direccion > velocidad_min_entrada):
                st.warning(f"⏳ **DECISIÓN: PREPARAR entrada en {row['ticker']}** (esperar más volumen o aceleración)")
            else:
                st.error(f"❌ **DECISIÓN: NO ENTRAR en {row['ticker']}** (condiciones insuficientes)")
    
    # Resumen de la mejor opción para entrar
    st.subheader("🎯 Recomendación final")
    
    mejor_timing = df_open.loc[df_open['Puntaje_timing_norm'].idxmax()]
    
    st.info(f"""
    **Según la simulación, el mejor candidato para entrar al open es: {mejor_timing['ticker']}**
    
    - Gap: {mejor_timing['gap_pct']*100:.2f}%
    - Velocidad vela 1: {mejor_timing['velocidad_vela1']*100:.2f}%/min
    - Aceleración: {mejor_timing['aceleracion']*100:.2f}%/min²
    - Volumen relativo: {mejor_timing['volumen_relativo_1']:.1f}x
    - Puntaje de timing: {mejor_timing['Puntaje_timing_norm']:.0%}
    """)
    
    # Gráfico comparativo de velocidades
    st.subheader("📊 Comparativa de velocidad y aceleración")
    
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gráfico de velocidades
    tickers_plot = df_open['ticker'].tolist()
    vel1 = df_open['velocidad_vela1'].values * 100
    vel2 = df_open['velocidad_vela2'].values * 100
    
    x = np.arange(len(tickers_plot))
    width = 0.35
    ax1.bar(x - width/2, vel1, width, label='Vela 1 (5 min)', color='steelblue')
    ax1.bar(x + width/2, vel2, width, label='Vela 2 (10 min)', color='coral')
    ax1.set_xlabel('Candidato')
    ax1.set_ylabel('Velocidad (%/min)')
    ax1.set_title('Velocidad de cambio en las primeras velas')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers_plot)
    ax1.legend()
    ax1.axhline(y=velocidad_min_entrada*100, color='green', linestyle='--', label=f'Umbral ({velocidad_min_entrada*100:.2f}%/min)')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de aceleración
    acel = df_open['aceleracion'].values * 100
    colors = ['green' if a*direccion > 0 else 'red' for a in df_open['aceleracion'].values]
    ax2.bar(tickers_plot, acel, color=colors, alpha=0.7)
    ax2.set_xlabel('Candidato')
    ax2.set_ylabel('Aceleración (%/min²)')
    ax2.set_title('Aceleración (positiva = gana fuerza)')
    ax2.axhline(y=0, color='black', linestyle='-')
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig2)

else:
    st.info("👆 Presiona 'Simular apertura de mercado' para ver cómo se comportarían los candidatos en las primeras 2 velas del open")

# ============================================================
# 6. DETALLE DE CADA CANDIDATO
# ============================================================

st.header("🔍 Detalle individual de cada candidato")

for i, row in df_ranking.iterrows():
    ticker = row['Ticker']
    sub = df[df['Ticker'] == ticker]
    
    with st.expander(f"{ticker} | k={row['k']:.3f} | R²={row['R²']:.2f} | Ímpetu={row['Ímpetu']:.2f}"):
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.plot(sub['Vela'], sub['EMA5'], 'o-', color='blue', label='Real (EMA5)', linewidth=2, markersize=4)
            ax2.plot(sub['Vela'], row['y_pred'], '--', color='red', label='Exponencial ajustada', linewidth=2)
            ax2.set_xlabel("Vela")
            ax2.set_ylabel("Votos Netos Suavizados")
            ax2.set_title(f"{ticker} - Curva exponencial")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
        
        with col_g2:
            st.metric("k (curvatura)", f"{row['k']:.4f}")
            st.metric("R² (ajuste)", f"{row['R²']:.2%}")
            st.metric("Longitud de arco", f"{row['Longitud_arco']:.2f}")
            st.metric("Velocidad final", f"{row['Velocidad_final']:.3f}")
            st.metric("Ímpetu", f"{row['Ímpetu']:.2f}")

# ============================================================
# 7. EXPORTAR DECISIÓN
# ============================================================

st.divider()
st.subheader("📤 Exportar decisión para tu estrategia")

if st.button("📋 Generar código para estrategia", use_container_width=True):
    mejor_fase_a = df_ranking.iloc[0]['Ticker']
    
    st.code(f"""
    # Código para integrar en tu estrategia PineScript / Python
    
    # === FASE A: CANDIDATO SELECCIONADO ===
    MEJOR_CANDIDATO = "{mejor_fase_a}"
    FACTOR_IMPETU = {df_ranking.iloc[0]['Ímpetu']:.2f}
    VELOCIDAD_FINAL = {df_ranking.iloc[0]['Velocidad_final']:.3f}
    CONFIANZA_ESTRUCTURAL = {df_ranking.iloc[0]['R²']:.2f}
    
    # === FASE B: CONDICIONES PARA ENTRAR AL OPEN ===
    # (estos valores se actualizarían en tiempo real)
    
    # Entrar en la segunda vela de 5 min si:
    # 1. Gap favorable en dirección del sesgo (>0.3%)
    # 2. Velocidad en primera vela > {velocidad_min_entrada*100:.2f}%/min
    # 3. Volumen en primera vela > {volumen_min_entrada}x promedio
    # 4. Aceleración positiva (gana fuerza)
    
    # Tamaño de posición sugerido:
    TAMANO_BASE = 1.0
    TAMANO_AJUSTADO = TAMANO_BASE * FACTOR_IMPETU * CONFIANZA_ESTRUCTURAL
    # Rango: {min(1.5, df_ranking.iloc[0]['Ímpetu'] * df_ranking.iloc[0]['R²']):.2f}x
    """, language="python")

st.caption("""
**Resumen del dashboard:**
- **Fase A:** Selecciona los mejores candidatos según ímpetu, velocidad y curvatura
- **Fase B:** Simula las primeras 2 velas del open para decidir timing de entrada
- **Decisión final:** Combina estructura (día anterior) + timing (intradía)
""")
