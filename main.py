
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import json
import random
from faker import Faker
import PyPDF2
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="MVP Diagnóstico Formación AI", layout="wide")

# Inicializar Faker para datos aleatorios
fake = Faker('es_ES')

# --- 1. SIMULACIÓN DE BASE DE DATOS (BACKEND) ---
@st.cache_data
def generate_mock_database():
    """Genera una base de datos simulada de expertos y recursos."""
    experts = []
    specialties = ['Liderazgo', 'Python & Data', 'Soft Skills', 'Agile', 'Ventas', 'Ciberseguridad']
    
    for i in range(15):
        experts.append({
            "id": f"EXP-{i+100}",
            "name": fake.name(),
            "specialty": random.choice(specialties),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "hourly_rate": random.randint(50, 200),
            "email": fake.email()
        })
    return pd.DataFrame(experts)

# Cargar base de datos simulada
experts_db = generate_mock_database()

# --- 2. FUNCIONES DE PROCESAMIENTO ---
def extract_text_from_pdf(uploaded_file):
    """Extrae texto de un archivo PDF cargado."""
    if uploaded_file is None:
        return ""
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_ai_diagnosis(api_key, pain_points, strategy_text):
    """Consulta a Groq para obtener el diagnóstico en formato JSON."""
    client = Groq(api_key=api_key)
    
    # Contexto para el modelo (Prompt Engineering)
    system_prompt = """
    Actúa como un Consultor Senior de RRHH experto en Formación Corporativa.
    Tu tarea es analizar los dolores de la empresa y su estrategia para generar un plan de formación.
    
    IMPORTANTE: Tu respuesta debe ser EXCLUSIVAMENTE un objeto JSON válido con la siguiente estructura, sin texto adicional antes o después:
    {
        "diagnosis_summary": "Resumen ejecutivo del problema detectado (max 50 palabras)",
        "identified_gaps": [
            {"gap": "Nombre de la brecha", "severity": score_1_to_10, "category": "Tecnica/Blanda/Estrategica"}
        ],
        "recommended_plan": [
            {"module": "Nombre del modulo", "duration": "Horas estimadas", "objective": "Objetivo de aprendizaje"}
        ],
        "recommended_specialties": ["Especialidad1", "Especialidad2"] 
    }
    """
    
    user_content = f"""
    DOLORES/NECESIDADES: {pain_points}
    
    CONTEXTO ESTRATÉGICO (PDF): {strategy_text[:2000]} (texto truncado para eficiencia)
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1,
            response_format={"type": "json_object"} # Forzamos modo JSON si está disponible o confiamos en el prompt
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        st.error(f"Error al conectar con Groq o procesar JSON: {e}")
        return None

# --- 3. INTERFAZ DE USUARIO (FRONTEND) ---

# Sidebar: Configuración
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=50)
st.sidebar.title("Configuración")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Ingresa tu API Key de Groq para activar el cerebro de IA.")
st.sidebar.markdown("---")
st.sidebar.info("Este MVP simula el flujo técnico de diagnóstico de formación automatizado.")

# Título Principal
st.title("🧠 Diagnóstico de Formación Corporativa con IA")
st.markdown("Sube tu estrategia y describe tus necesidades para generar un plan de capacitación automático.")

# Columnas para Inputs
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Entradas (Inputs)")
    pain_points = st.text_area("Dolores / Necesidades de Formación", 
                               placeholder="Ej: El equipo de ventas no está cerrando tratos digitales, falta liderazgo en mandos medios...",
                               height=150)
    
    uploaded_file = st.file_uploader("Cargar PDF de Estrategia (Opcional)", type="pdf")
    
    process_btn = st.button("Generar Diagnóstico 🚀", type="primary", use_container_width=True)

# Lógica Principal al hacer clic
if process_btn:
    if not groq_api_key:
        st.warning("⚠️ Por favor ingresa tu API Key de Groq en la barra lateral.")
    elif not pain_points:
        st.warning("⚠️ Por favor describe los dolores o necesidades.")
    else:
        with st.spinner("🔍 Analizando documentos, consultando a Llama-3 y buscando expertos..."):
            # 1. Extracción
            pdf_text = extract_text_from_pdf(uploaded_file) if uploaded_file else "No se proporcionó PDF, basar solo en dolores."
            
            # 2. IA Engine
            diagnosis_result = get_ai_diagnosis(groq_api_key, pain_points, pdf_text)
            
            if diagnosis_result:
                # Guardar en estado de sesión para no perderlo al refrescar filtros
                st.session_state['diagnosis'] = diagnosis_result
                st.session_state['processed'] = True

# --- 4. DASHBOARD DE RESULTADOS ---
if st.session_state.get('processed'):
    data = st.session_state['diagnosis']
    
    st.markdown("---")
    st.subheader("2. Dashboard de Resultados")
    
    # Métricas y Resumen
    m1, m2, m3 = st.columns(3)
    m1.metric("Brechas Identificadas", len(data['identified_gaps']))
    m2.metric("Módulos Sugeridos", len(data['recommended_plan']))
    m3.metric("Severidad Promedio", round(sum(d['severity'] for d in data['identified_gaps'])/len(data['identified_gaps']), 1))
    
    st.info(f"**Diagnóstico:** {data['diagnosis_summary']}")
    
    # Visualizaciones Dinámicas
    c_chart1, c_chart2 = st.columns([1, 1])
    
    with c_chart1:
        st.markdown("#### Mapa de Brechas (Radar)")
        df_gaps = pd.DataFrame(data['identified_gaps'])
        
        # Gráfico Radar
        if not df_gaps.empty:
            fig_radar = px.line_polar(df_gaps, r='severity', theta='gap', line_close=True, 
                                      title="Severidad de Brechas Detectadas",
                                      template="plotly_dark")
            fig_radar.update_traces(fill='toself')
            st.plotly_chart(fig_radar, use_container_width=True)
            
    with c_chart2:
        st.markdown("#### Plan de Formación Sugerido")
        df_plan = pd.DataFrame(data['recommended_plan'])
        st.dataframe(df_plan, hide_index=True, use_container_width=True)

    # --- FILTRADO Y RECOMENDACIÓN DE EXPERTOS ---
    st.markdown("---")
    st.subheader("3. Expertos Recomendados & Agendamiento")
    
    # Lógica de match (simple string matching con las especialidades recomendadas por la IA)
    rec_specialties = data.get('recommended_specialties', [])
    
    # Filtro visual para el usuario
    selected_filter = st.selectbox("Filtrar Expertos por Especialidad Sugerida:", ["Todos"] + rec_specialties)
    
    if selected_filter != "Todos":
        # Filtramos simulando búsqueda semántica simple
        filtered_experts = experts_db[experts_db['specialty'].str.contains(selected_filter, case=False, na=False)]
        # Si no hay match exacto, mostramos random para demo
        if filtered_experts.empty:
             filtered_experts = experts_db.sample(3)
    else:
        # Si es "Todos", mostramos una mezcla basada en las recomendaciones
        filtered_experts = experts_db[experts_db['specialty'].isin(rec_specialties)]
        if filtered_experts.empty:
            filtered_experts = experts_db.sample(5)

    # Mostrar tarjetas de expertos
    for index, row in filtered_experts.iterrows():
        with st.expander(f"🎓 {row['name']} - Especialista en {row['specialty']}"):
            ec1, ec2 = st.columns([3, 1])
            with ec1:
                st.write(f"**ID:** {row['id']}")
                st.write(f"**Valoración:** ⭐ {row['rating']}/5.0")
                st.write(f"**Tarifa:** ${row['hourly_rate']}/hora")
            with ec2:
                if st.button(f"Agendar con {row['name'].split()[0]}", key=row['id']):
                    # Simulación de Disparador de Evento (Email/Calendar)
                    st.success(f"✅ Solicitud enviada a {row['email']}. Se ha creado el evento en Google Calendar.")
                    st.balloons()
