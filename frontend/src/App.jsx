import { useState, useEffect, useRef, useCallback } from "react";

// ══════════════════════════════════════════════════════════════════════════════
//  DATOS DEL PROYECTO — Edita aquí los nombres e integrantes
// ══════════════════════════════════════════════════════════════════════════════
const PROYECTO = {
  universidad: "UNIVERSIDAD MARIANO GÁLVEZ DE GUATEMALA",
  facultad:    "Facultad de Ingeniería en Sistemas",
  catedratico: "Catedrático: Ing. Jorge Luis Salguero Galicia",
  curso:       "Inteligencia Artificial",
  titulo:      "Sistema de Predicción de Series de Tiempo con Redes LSTM",
  integrantes: [
    { nombre: "José Andrés López Flores",            carnet: "3590-22-6856"  },
    { nombre: "Dieter Randolfo Osorio Hernandez",    carnet: "3590-21-19249" },
    { nombre: "Jonathan Ananías Velasquez Martínez", carnet: "3590-22-2636"  },
    { nombre: "Daylin Rebeca Temaj Perez",           carnet: "3590-22-17218" },
    { nombre: "Kateryn Yohana Pasán De León",        carnet: "3590-22-9755"  },
  ],
};

// ══════════════════════════════════════════════════════════════════════════════
//  ╔══════════════════════════════════════════════════════════════════════╗
//  ║                  ZONA DE DISEÑO — PORTADA                          ║
//  ║  Todo lo que está aquí controla el aspecto de la pantalla inicial.  ║
//  ║  Puedes cambiar colores, edificios y fondos sin tocar la lógica.   ║
//  ╚══════════════════════════════════════════════════════════════════════╝

// Imágenes (ponlas en frontend/public/img/)
const IMGS = {
  logo:      "/img/logo.png",
  bgPortada: "/img/bg_portada.png",   // si existe, reemplaza el degradado
};

// Degradado del cielo en la portada
// Para usar imagen: cambia esto por `url(${IMGS.bgPortada}) center/cover`
const FONDO_PORTADA = `
  radial-gradient(ellipse at 20% 80%, rgba(56,189,248,.15) 0%, transparent 50%),
  radial-gradient(ellipse at 80% 20%, rgba(245,158,11,.10) 0%, transparent 50%),
  linear-gradient(180deg, #050810 0%, #0a1628 50%, #0d1f3c 100%)
`;

// Edificios del skyline — { x: posición, w: ancho, h: altura }
// Agregar, quitar o modificar aquí para cambiar la ciudad
const EDIFICIOS = [
  {x:0,   w:60,  h:80 }, {x:55,  w:40,  h:120}, {x:90,  w:80,  h:60 },
  {x:165, w:50,  h:100}, {x:210, w:120, h:80 }, {x:325, w:60,  h:140},
  {x:380, w:90,  h:90 }, {x:465, w:50,  h:110}, {x:510, w:70,  h:70 },
  {x:575, w:100, h:120}, {x:670, w:60,  h:90 }, {x:725, w:80,  h:60 },
  {x:800, w:50,  h:130}, {x:845, w:90,  h:80 }, {x:930, w:60,  h:100},
];
//  ╚══════════════════════════════════════════════════════════════════════╝


// ══════════════════════════════════════════════════════════════════════════════
//  CONFIGURACIÓN TÉCNICA
// ══════════════════════════════════════════════════════════════════════════════

const API = "http://localhost:5000";

// Agentes disponibles por dataset
// Para agregar uno nuevo (ej: Bidirectional LSTM), solo agrega una línea aquí
const AGENTES_POR_DATASET = {
  sismos: [
    { id: "lstm_simple_sismos",   label: "LSTM Simple",   descripcion: "Una sola capa LSTM. Más rápido, ideal para empezar." },
    { id: "stacked_lstm_sismos",  label: "Stacked LSTM",  descripcion: "Tres capas apiladas. Detecta patrones más complejos." },
    // { id: "bilstm_sismos", label: "Bidirectional LSTM", descripcion: "Lee la secuencia en ambas direcciones." },
  ],
  energia: [
    { id: "lstm_simple_energia",  label: "LSTM Simple",   descripcion: "Una sola capa LSTM. Más rápido, ideal para empezar." },
    { id: "stacked_lstm_energia", label: "Stacked LSTM",  descripcion: "Tres capas apiladas. Detecta patrones más complejos." },
    // { id: "bilstm_energia", label: "Bidirectional LSTM", descripcion: "Lee la secuencia en ambas direcciones." },
  ],
};

// Rangos reales del dataset de sismos (para desnormalizar el pronóstico)
const SISMOS_MAG_MIN = 2.7;
const SISMOS_MAG_MAX = 6.9;

// Info de fallback si la API no responde
const FALLBACK_INFO = {
  energia: {
    nombre: "Consumo Eléctrico Doméstico", fuente: "UCI ML Repository",
    registros: "2,075,259 mediciones", periodo: "Dic 2006 — Nov 2010",
    frecuencia: "1 medición por minuto", window_size: 60, color: "#F59E0B",
    features: ["Global_active_power","Global_reactive_power","Voltage","Global_intensity","Sub_metering_1","Sub_metering_2","Sub_metering_3"],
    target: "Global_active_power (potencia activa en kW)",
    descripcion: "Mediciones de consumo eléctrico de un hogar en Francia. Permite predecir la demanda energética para optimizar redes eléctricas inteligentes.",
  },
  sismos: {
    nombre: "Actividad Sísmica — Guatemala", fuente: "USGS Earthquake Catalog",
    registros: "3,586 eventos", periodo: "1970 — 2026",
    frecuencia: "Serie diaria agregada", window_size: 14, color: "#EF4444",
    features: ["avg_mag","max_mag","avg_depth","count_sismos","hay_actividad"],
    target: "avg_mag (magnitud promedio diaria)",
    descripcion: "Registro de actividad sísmica en Centroamérica. Útil para analizar patrones para sistemas de monitoreo en Guatemala.",
  },
};

// ══════════════════════════════════════════════════════════════════════════════
//  ESTILOS GLOBALES
// ══════════════════════════════════════════════════════════════════════════════
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&family=Rajdhani:wght@400;600;700&display=swap');
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
  :root{
    --c-bg:#0a0e1a; --c-dark:#0d1220; --c-border:#2a3a5c;
    --c-energy:#F59E0B; --c-seismic:#EF4444; --c-accent:#38BDF8;
    --c-green:#4ADE80; --c-white:#F1F5F9; --c-gray:#64748B;
    --font-px:'Press Start 2P',monospace; --font-vt:'VT323',monospace; --font-body:'Rajdhani',sans-serif;
  }
  html,body,#root{width:100%;height:100%;background:var(--c-bg);color:var(--c-white);font-family:var(--font-body);overflow:hidden;}
  .px-btn{font-family:var(--font-px);font-size:10px;letter-spacing:1px;cursor:pointer;border:2px solid;padding:10px 20px;transition:transform .05s;text-transform:uppercase;background:transparent;}
  .px-btn:active{transform:translate(2px,2px);}
  .px-btn:disabled{opacity:.5;cursor:not-allowed;}
  .scanlines::before{content:'';position:absolute;inset:0;background:repeating-linear-gradient(to bottom,transparent 0,transparent 3px,rgba(0,0,0,.08) 3px,rgba(0,0,0,.08) 4px);pointer-events:none;z-index:10;}
  .screen-enter{animation:sIn .4s steps(8,end) forwards;}
  @keyframes sIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
  .blink{animation:blink 1s steps(1) infinite;}
  @keyframes blink{50%{opacity:0}}
  .float{animation:float 3s ease-in-out infinite;}
  @keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
  .glitch{animation:glitch 4s steps(1) infinite;}
  @keyframes glitch{0%,94%,100%{text-shadow:none}95%{text-shadow:-2px 0 #EF4444,2px 0 #38BDF8}97%{text-shadow:2px 0 #F59E0B,-2px 0 #4ADE80}}
  .px-grid{background-image:linear-gradient(rgba(56,189,248,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(56,189,248,.03) 1px,transparent 1px);background-size:32px 32px;}
  ::-webkit-scrollbar{width:4px;} ::-webkit-scrollbar-track{background:var(--c-dark);} ::-webkit-scrollbar-thumb{background:var(--c-border);}
  input,select{outline:none;}
`;


// ══════════════════════════════════════════════════════════════════════════════
//  PANTALLA 1 — PORTADA
// ══════════════════════════════════════════════════════════════════════════════
function ScreenPortada({ onNext }) {
  return (
    <div className="scanlines screen-enter" style={{
      width:"100vw", height:"100vh", position:"relative", overflow:"hidden",
      background: FONDO_PORTADA,
    }}>
      <PixelStars />
      <PixelCity />
      <div style={{position:"absolute",inset:0,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",gap:"14px",padding:"20px",zIndex:5}}>
        <div className="float">
          <img src={IMGS.logo} alt="Logo UMG" onError={e => e.target.style.display="none"}
            style={{width:"80px",height:"80px",imageRendering:"pixelated",filter:"drop-shadow(0 0 12px rgba(56,189,248,.5))"}} />
        </div>
        <div style={{fontFamily:"var(--font-px)",fontSize:"8px",color:"var(--c-accent)",textAlign:"center",lineHeight:"1.8",letterSpacing:"2px",textShadow:"0 0 20px rgba(56,189,248,.6)"}}>
          {PROYECTO.universidad}
        </div>
        <div style={{fontFamily:"var(--font-vt)",fontSize:"17px",color:"#94a3b8",textAlign:"center",lineHeight:"1.6"}}>
          <div>{PROYECTO.facultad}</div>
          <div>{PROYECTO.catedratico}</div>
          <div style={{color:"var(--c-energy)"}}>{PROYECTO.curso}</div>
        </div>
        <div className="glitch" style={{fontFamily:"var(--font-px)",fontSize:"10px",color:"var(--c-white)",textAlign:"center",lineHeight:"2",maxWidth:"680px",textShadow:"0 0 30px rgba(255,255,255,.3)"}}>
          {PROYECTO.titulo}
        </div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"3px 28px",background:"rgba(17,24,39,.85)",border:"1px solid var(--c-border)",padding:"10px 20px"}}>
          {PROYECTO.integrantes.map((p, i) => (
            <div key={i} style={{fontFamily:"var(--font-vt)",fontSize:"16px",display:"flex",justifyContent:"space-between",gap:"20px",color:"#cbd5e1"}}>
              <span>{p.nombre}</span>
              <span style={{color:"var(--c-accent)",whiteSpace:"nowrap"}}>{p.carnet}</span>
            </div>
          ))}
        </div>
        <button className="px-btn" onClick={onNext}
          style={{marginTop:"10px",background:"var(--c-accent)",borderColor:"#0ea5e9",color:"#0a0e1a",fontSize:"10px",padding:"12px 32px",boxShadow:"0 0 20px rgba(56,189,248,.4)"}}>
          ▶ ¡EMPECEMOS! <span className="blink">_</span>
        </button>
      </div>
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════════════════
//  PANTALLA 2 — SELECCIÓN DE DATASET
// ══════════════════════════════════════════════════════════════════════════════
function ScreenSeleccion({ onSelect }) {
  const [hover, setHover] = useState(null);
  const [info, setInfo]   = useState({ energia: FALLBACK_INFO.energia, sismos: FALLBACK_INFO.sismos });

  useEffect(() => {
    ["energia","sismos"].forEach(async ds => {
      try {
        const r = await fetch(`${API}/api/dataset-info/${ds}`);
        if (r.ok) {
          const data = await r.json();
          setInfo(prev => ({ ...prev, [ds]: data }));
        }
      } catch {}
    });
  }, []);

  const opciones = [
    { key: "energia", label: "⚡ ENERGÍA ELÉCTRICA", color: "var(--c-energy)", raw: "#F59E0B" },
    { key: "sismos",  label: "🌍 ACTIVIDAD SÍSMICA",  color: "var(--c-seismic)", raw: "#EF4444" },
  ];

  return (
    <div className="screen-enter" style={{width:"100vw",height:"100vh",display:"flex",flexDirection:"column",background:"var(--c-bg)"}}>
      <div style={{textAlign:"center",padding:"14px",borderBottom:"2px solid var(--c-border)",background:"var(--c-dark)"}}>
        <span style={{fontFamily:"var(--font-px)",fontSize:"9px",color:"var(--c-white)",letterSpacing:"2px"}}>SELECCIONA TU DATASET</span>
      </div>
      <div style={{flex:1,display:"flex",overflow:"hidden"}}>
        {opciones.map((op, idx) => (
          <div key={op.key}
            onMouseEnter={() => setHover(op.key)}
            onMouseLeave={() => setHover(null)}
            style={{
              flex:1, display:"flex", flexDirection:"column",
              borderRight: idx === 0 ? "2px solid var(--c-border)" : "none",
              overflow:"hidden", transition:"all .2s",
              background: hover === op.key
                ? `linear-gradient(180deg,${op.raw}18 0%,var(--c-dark) 100%)`
                : "var(--c-dark)",
            }}>
            <div style={{padding:"18px 22px 10px",fontFamily:"var(--font-px)",fontSize:"11px",color:op.color,letterSpacing:"1px",
              textShadow: hover===op.key ? `0 0 20px ${op.raw}80` : "none",transition:"text-shadow .2s"}}>
              {op.label}
            </div>
            <div style={{flex:1,padding:"0 22px",overflowY:"auto"}}>
              <DatasetCard info={info[op.key]} color={op.color} raw={op.raw} />
            </div>
            <div style={{padding:"16px 22px"}}>
              <button className="px-btn" onClick={() => onSelect(op.key)}
                style={{width:"100%",background:hover===op.key?op.color:"transparent",borderColor:op.raw,
                  color:hover===op.key?"#0a0e1a":op.color,fontSize:"9px",transition:"all .15s"}}>
                INICIAR →
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function DatasetCard({ info, color, raw }) {
  if (!info) return null;
  return (
    <div style={{display:"flex",flexDirection:"column",gap:"10px"}}>
      <p style={{fontFamily:"var(--font-body)",fontSize:"15px",color:"#94a3b8",lineHeight:"1.6"}}>{info.descripcion}</p>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"6px"}}>
        {[{l:"REGISTROS",v:info.registros},{l:"PERÍODO",v:info.periodo},{l:"FRECUENCIA",v:info.frecuencia},{l:"VENTANA",v:`${info.window_size} pasos`}].map((s,i)=>(
          <div key={i} style={{background:"#0d1220",border:`1px solid ${raw}40`,padding:"7px 10px"}}>
            <div style={{fontFamily:"var(--font-px)",fontSize:"7px",color,marginBottom:"3px"}}>{s.l}</div>
            <div style={{fontFamily:"var(--font-vt)",fontSize:"17px",color:"#e2e8f0"}}>{s.v}</div>
          </div>
        ))}
      </div>
      <div>
        <div style={{fontFamily:"var(--font-px)",fontSize:"7px",color,marginBottom:"6px"}}>FEATURES DEL MODELO</div>
        <div style={{display:"flex",flexWrap:"wrap",gap:"5px"}}>
          {info.features?.map((f,i) => (
            <span key={i} style={{fontFamily:"var(--font-vt)",fontSize:"15px",background:`${raw}20`,color:"#cbd5e1",padding:"1px 8px",border:`1px solid ${raw}40`}}>{f}</span>
          ))}
        </div>
      </div>
      <div style={{background:`${raw}15`,border:`1px solid ${raw}50`,padding:"8px 12px"}}>
        <div style={{fontFamily:"var(--font-px)",fontSize:"7px",color,marginBottom:"3px"}}>VARIABLE OBJETIVO</div>
        <div style={{fontFamily:"var(--font-vt)",fontSize:"19px",color:"#f1f5f9"}}>{info.target}</div>
      </div>
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════════════════
//  PANTALLA 3 — ENTRENAMIENTO Y RESULTADOS
// ══════════════════════════════════════════════════════════════════════════════
function ScreenEntrenamiento({ dataset, onBack }) {
  const isEnergia = dataset === "energia";
  const accent    = isEnergia ? "var(--c-energy)" : "var(--c-seismic)";
  const raw       = isEnergia ? "#F59E0B" : "#EF4444";

  const agentes   = AGENTES_POR_DATASET[dataset] || [];

  // ── Estado principal ────────────────────────────────────────────────────────
  const [agenteId,   setAgenteId]   = useState(agentes[0]?.id || "");
  const [config,     setConfig]     = useState({ epochs: isEnergia?10:50, batch_size:64, model_name: agentes[0]?.id || "" });
  const [progress,   setProgress]   = useState(null);
  const [results,    setResults]    = useState(null);
  const [polling,    setPolling]    = useState(false);
  const [tab,        setTab]        = useState("proceso");
  const [cargando,   setCargando]   = useState(false);
  const [error,      setError]      = useState(null);
  const [modelos,    setModelos]    = useState([]);
  const [modeloSel,  setModeloSel]  = useState("");

  const logRef      = useRef(null);
  const intervalRef = useRef(null);

  // Cuando cambia el agente, actualizar el nombre sugerido del archivo
  useEffect(() => {
    const agente = agentes.find(a => a.id === agenteId);
    if (agente) setConfig(prev => ({ ...prev, model_name: agente.id }));
  }, [agenteId]);

  // Cargar lista de modelos disponibles
  const refrescarModelos = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/modelos`);
      if (!r.ok) return;
      const lista = await r.json();
      // Mostrar solo modelos del dataset actual
      const disponibles = lista.filter(m => m.dataset === dataset && m.entrenado);
      setModelos(disponibles);
      setModeloSel(prev => {
        if (disponibles.length === 0) return "";
        if (disponibles.find(m => m.nombre === prev)) return prev;
        return disponibles[0].nombre;
      });
    } catch {}
  }, [dataset]);

  useEffect(() => { refrescarModelos(); }, []);

  // Auto-scroll del log
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [progress?.log]);

  // Polling del progreso
  useEffect(() => {
    if (!polling) return;
    intervalRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${API}/api/entrenar/progreso`);
        const d = await r.json();
        setProgress(d);
        if (d.terminado) {
          setPolling(false);
          clearInterval(intervalRef.current);
          if (!d.error) {
            setResults({ metricas: d.metricas, tabla_pred: d.tabla_pred, pronostico: d.pronostico, dataset_info: d.dataset_info });
            setTab("resultados");
            refrescarModelos();
          }
        }
      } catch {}
    }, 800);
    return () => clearInterval(intervalRef.current);
  }, [polling]);

  // Iniciar entrenamiento
  async function handleEntrenar() {
    setResults(null); setProgress(null); setError(null); setTab("proceso");
    const agente = agentes.find(a => a.id === agenteId);
    try {
      const r = await fetch(`${API}/api/entrenar`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset, ...config, tipo: agente?.label || "LSTM Simple" }),
      });
      const d = await r.json();
      if (d.error) { setError(d.error); return; }
      setPolling(true);
    } catch {
      setError("No se pudo conectar con la API en localhost:5000. ¿Está corriendo api.py?");
    }
  }

  // Cargar modelo existente
  async function handleCargar() {
    if (!modeloSel) { setError("Selecciona un modelo de la lista."); return; }
    setCargando(true); setError(null); setResults(null);
    try {
      const r = await fetch(`${API}/api/predecir`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modelo: modeloSel }),
      });
      const d = await r.json();
      if (d.error) { setError(d.error); setCargando(false); return; }
      setResults({ metricas: d.metricas, tabla_pred: d.tabla_pred, pronostico: d.pronostico, dataset_info: d.dataset_info });
      setProgress({ log: [`Modelo cargado: ${modeloSel}`, `Tipo: ${d.info_modelo?.tipo || "—"}`, `RMSE: ${d.metricas?.rmse}  MAE: ${d.metricas?.mae}  MAPE: ${d.metricas?.mape}%`], terminado: true, corriendo: false });
      setTab("resultados");
    } catch {
      setError("Error de conexión. Verifica que la API esté corriendo.");
    } finally {
      setCargando(false);
    }
  }

  const pct      = progress ? Math.round(((progress.epoca_actual||0) / (progress.total_epocas||1)) * 100) : 0;
  const corriendo = progress?.corriendo;
  const tabs     = ["proceso","resultados","predicciones", ...(dataset==="sismos"?["pronostico"]:[])];

  return (
    <div className="screen-enter px-grid" style={{width:"100vw",height:"100vh",display:"flex",overflow:"hidden",background:"var(--c-bg)"}}>

      {/* ── PANEL IZQUIERDO ────────────────────────────────────── */}
      <div style={{width:"285px",minWidth:"285px",background:"var(--c-dark)",borderRight:`2px solid ${raw}50`,display:"flex",flexDirection:"column",overflowY:"auto"}}>

        {/* Header */}
        <div style={{padding:"14px",borderBottom:`2px solid ${raw}30`,background:`${raw}12`}}>
          <button onClick={onBack} style={{fontFamily:"var(--font-px)",fontSize:"7px",background:"transparent",border:"none",color:"var(--c-gray)",cursor:"pointer",marginBottom:"6px"}}>
            ← VOLVER
          </button>
          <div style={{fontFamily:"var(--font-px)",fontSize:"8px",color:accent,lineHeight:"1.8"}}>
            {isEnergia ? "⚡ ENERGÍA ELÉCTRICA" : "🌍 ACTIVIDAD SÍSMICA"}
          </div>
        </div>

        <div style={{padding:"14px",display:"flex",flexDirection:"column",gap:"16px"}}>

          {/* Error */}
          {error && (
            <div style={{background:"rgba(239,68,68,.15)",border:"1px solid #EF4444",padding:"10px",fontFamily:"var(--font-vt)",fontSize:"15px",color:"#EF4444",lineHeight:"1.4"}}>
              {error}
              <button onClick={() => setError(null)} style={{display:"block",marginTop:"6px",background:"transparent",border:"none",color:"#64748b",cursor:"pointer",fontFamily:"var(--font-vt)",fontSize:"14px"}}>
                [cerrar]
              </button>
            </div>
          )}

          {/* PASO 1 — Elegir agente */}
          <Label text="PASO 1 — TIPO DE AGENTE" color={accent} />
          <div style={{display:"flex",flexDirection:"column",gap:"8px"}}>
            {agentes.map(a => (
              <div key={a.id} onClick={() => setAgenteId(a.id)} style={{
                padding:"10px 12px", cursor:"pointer", transition:"all .15s",
                border:`2px solid ${agenteId===a.id ? raw : raw+"40"}`,
                background: agenteId===a.id ? `${raw}18` : "transparent",
              }}>
                <div style={{fontFamily:"var(--font-px)",fontSize:"8px",marginBottom:"4px",color:agenteId===a.id?accent:"#94a3b8"}}>
                  {agenteId===a.id?"▶ ":""}{a.label}
                </div>
                <div style={{fontFamily:"var(--font-vt)",fontSize:"14px",color:"#475569",lineHeight:"1.4"}}>{a.descripcion}</div>
              </div>
            ))}
          </div>

          {/* PASO 2 — Configurar */}
          <Label text="PASO 2 — CONFIGURACIÓN" color={accent} />
          <Campo label="ÉPOCAS" type="number" min={1} max={200} value={config.epochs}
            onChange={v => setConfig(p => ({...p, epochs:+v}))} />
          <Campo label="BATCH SIZE" type="number" value={config.batch_size}
            onChange={v => setConfig(p => ({...p, batch_size:+v}))} />
          <Campo label="NOMBRE DEL ARCHIVO (.pth)" type="text" value={config.model_name}
            onChange={v => setConfig(p => ({...p, model_name:v}))} />

          {/* PASO 3 — Entrenar */}
          <button className="px-btn" onClick={handleEntrenar} disabled={corriendo}
            style={{background:corriendo?raw:accent,borderColor:raw,color:"#0a0e1a",width:"100%",fontSize:"8px"}}>
            {corriendo ? "⟳ ENTRENANDO..." : "▶ PASO 3 — ENTRENAR"}
          </button>

          <div style={{borderTop:`1px solid ${raw}20`}} />

          {/* Cargar modelo */}
          <Label text="CARGAR MODELO (.pth)" color={accent} />
          {modelos.length === 0 ? (
            <div style={{fontFamily:"var(--font-vt)",fontSize:"15px",color:"#475569",lineHeight:"1.6"}}>
              Todavía no hay modelos guardados para este dataset.<br/>
              Entrena uno con los pasos de arriba, o corre los scripts de agents/.
            </div>
          ) : (
            <>
              <div style={{fontFamily:"var(--font-vt)",fontSize:"13px",color:"#475569"}}>
                Modelos entrenados disponibles:
              </div>
              <select value={modeloSel} onChange={e => setModeloSel(e.target.value)}
                style={{width:"100%",background:"#0d1220",color:"#e2e8f0",border:`1px solid ${raw}50`,fontFamily:"var(--font-vt)",fontSize:"16px",padding:"5px"}}>
                <option value="">-- elige uno --</option>
                {modelos.map(m => (
                  <option key={m.nombre} value={m.nombre}>
                    {m.tipo === "Modelo Custom" ? m.nombre : m.tipo} — {m.nombre} ({m.tamano_kb}KB)
                  </option>
                ))}
              </select>
              <button className="px-btn" onClick={handleCargar} disabled={cargando||!modeloSel}
                style={{width:"100%",borderColor:raw,color:accent,fontSize:"8px"}}>
                {cargando ? "⟳ CARGANDO..." : "CARGAR Y EVALUAR"}
              </button>
              <button onClick={refrescarModelos}
                style={{background:"transparent",border:"none",color:"#475569",cursor:"pointer",fontFamily:"var(--font-vt)",fontSize:"14px",textAlign:"left"}}>
                ↻ refrescar lista
              </button>
            </>
          )}

        </div>
      </div>

      {/* ── PANEL DERECHO ──────────────────────────────────────── */}
      <div style={{flex:1,display:"flex",flexDirection:"column",overflow:"hidden"}}>

        {/* Tabs */}
        <div style={{display:"flex",borderBottom:"2px solid var(--c-border)",background:"var(--c-dark)"}}>
          {tabs.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              fontFamily:"var(--font-px)",fontSize:"7px",padding:"12px 16px",border:"none",cursor:"pointer",
              letterSpacing:"1px",textTransform:"uppercase",
              background: tab===t ? `${raw}20` : "transparent",
              color: tab===t ? accent : "var(--c-gray)",
              borderBottom: tab===t ? `2px solid ${raw}` : "2px solid transparent",
            }}>
              {t==="proceso"?"📟 PROCESO":t==="resultados"?"📊 RESULTADOS":t==="predicciones"?"🔢 PREDICCIONES":"📅 PRONÓSTICO"}
            </button>
          ))}
        </div>

        {/* Contenido de tabs */}
        <div style={{flex:1,overflowY:"auto",padding:"18px",display:"flex",flexDirection:"column",gap:"14px"}}>

          {/* ── PROCESO ── */}
          {tab === "proceso" && (
            <>
              {progress && (
                <div style={{background:"var(--c-dark)",border:`1px solid ${raw}40`,padding:"14px"}}>
                  <div style={{display:"flex",justifyContent:"space-between",marginBottom:"8px"}}>
                    <span style={{fontFamily:"var(--font-px)",fontSize:"8px",color:accent}}>
                      {corriendo ? `ÉPOCA ${progress.epoca_actual} / ${progress.total_epocas}` : "COMPLETADO"}
                    </span>
                    <span style={{fontFamily:"var(--font-px)",fontSize:"8px",color:"var(--c-green)"}}>{pct}%</span>
                  </div>
                  <div style={{width:"100%",height:"18px",background:"#1e293b",border:"2px solid #334155",position:"relative",overflow:"hidden"}}>
                    <div style={{height:"100%",width:`${pct}%`,background:`linear-gradient(90deg,${raw},${raw}aa)`,transition:"width .3s steps(20,end)"}} />
                    <div style={{position:"absolute",inset:0,backgroundImage:`repeating-linear-gradient(90deg,transparent,transparent 7px,rgba(0,0,0,.15) 7px,rgba(0,0,0,.15) 8px)`,pointerEvents:"none"}} />
                  </div>
                  {corriendo && (
                    <div style={{display:"flex",gap:"20px",marginTop:"8px",fontFamily:"var(--font-vt)",fontSize:"17px"}}>
                      <span>Train: <span style={{color:"var(--c-green)"}}>{progress.loss_train}</span></span>
                      <span>Val: <span style={{color:accent}}>{progress.loss_val}</span></span>
                    </div>
                  )}
                </div>
              )}
              <div style={{background:"#050810",border:`1px solid ${raw}30`,flex:1,minHeight:"300px",display:"flex",flexDirection:"column"}}>
                <div style={{padding:"8px 12px",borderBottom:`1px solid ${raw}20`,fontFamily:"var(--font-px)",fontSize:"7px",color:accent}}>
                  CONSOLA — LOG EN TIEMPO REAL
                </div>
                <div ref={logRef} style={{flex:1,overflowY:"auto",padding:"10px 14px",fontFamily:"var(--font-vt)",fontSize:"16px",lineHeight:"1.6",color:"#64748b",maxHeight:"420px"}}>
                  {progress?.log?.length > 0
                    ? progress.log.map((l,i) => (
                        <div key={i} style={{color:l.includes("mejor")?"var(--c-green)":l.includes("ERROR")?"var(--c-seismic)":l.includes("guardado")?"var(--c-green)":l.includes("Early")?"var(--c-energy)":"#64748b",whiteSpace:"pre"}}>
                          {l}
                        </div>
                      ))
                    : <div style={{color:"#334155"}}>Presiona "PASO 3 — ENTRENAR" para empezar<span className="blink">_</span></div>
                  }
                </div>
              </div>
            </>
          )}

          {/* ── RESULTADOS ── */}
          {tab === "resultados" && (
            <>
              {!results && <Vacio accent={accent} msg="Entrena o carga un modelo para ver los resultados." />}
              {results?.metricas && (
                <>
                  <Titulo text="¿QUÉ TAN BIEN PREDICE EL MODELO?" color={accent} />
                  <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:"10px"}}>
                    {[
                      {
                        label: "RMSE",
                        val:   results.metricas.rmse,
                        titulo: "¿Qué tan lejos estuvo el modelo?",
                        explicacion: `En promedio, cada predicción se alejó ${results.metricas.rmse} unidades del valor real (escala 0 a 1). Los errores grandes pesan más en este cálculo.`,
                        estado: results.metricas.rmse < 0.05 ? "EXCELENTE" : results.metricas.rmse < 0.10 ? "ACEPTABLE" : "MEJORABLE",
                        color: "var(--c-accent)",
                      },
                      {
                        label: "MAE",
                        val:   results.metricas.mae,
                        titulo: "Error promedio por predicción",
                        explicacion: `En cada intento individual, el modelo se equivocó por ${results.metricas.mae} unidades. Aquí todos los errores cuentan igual, sin importar su tamaño.`,
                        estado: results.metricas.mae < 0.03 ? "EXCELENTE" : results.metricas.mae < 0.07 ? "ACEPTABLE" : "MEJORABLE",
                        color: "var(--c-green)",
                      },
                      {
                        label: "MAPE",
                        val:   `${results.metricas.mape}%`,
                        titulo: "¿Cuánto se equivocó en porcentaje?",
                        explicacion: `El modelo se equivocó en promedio un ${results.metricas.mape}% respecto al valor real. Por ejemplo, si la magnitud era 4.5, predijo algo entre ${(4.5*(1-results.metricas.mape/100)).toFixed(2)} y ${(4.5*(1+results.metricas.mape/100)).toFixed(2)}.`,
                        estado: results.metricas.mape < 10 ? "ACEPTABLE" : results.metricas.mape < 20 ? "REGULAR" : "MEJORABLE",
                        color: accent,
                      },
                    ].map((m, i) => (
                      <div key={i} style={{background:"var(--c-dark)",border:`2px solid ${raw}50`,padding:"14px"}}>
                        <div style={{fontFamily:"var(--font-px)",fontSize:"7px",color:m.color,marginBottom:"6px"}}>{m.label}</div>
                        <div style={{fontFamily:"var(--font-px)",fontSize:"18px",color:"var(--c-white)",marginBottom:"8px"}}>{m.val}</div>
                        <div style={{fontFamily:"var(--font-vt)",fontSize:"16px",color:m.color,marginBottom:"6px"}}>{m.titulo}</div>
                        <div style={{fontFamily:"var(--font-body)",fontSize:"13px",color:"#64748b",lineHeight:"1.5",marginBottom:"10px"}}>{m.explicacion}</div>
                        <span style={{
                          fontFamily:"var(--font-px)",fontSize:"9px",padding:"4px 10px",
                          color:"#0a0e1a",
                          background: m.estado==="EXCELENTE"?"var(--c-green)":m.estado==="ACEPTABLE"?"var(--c-energy)":m.estado==="REGULAR"?"#f97316":"var(--c-seismic)",
                        }}>{m.estado}</span>
                      </div>
                    ))}
                  </div>

                  {results.dataset_info && (
                    <>
                      <Titulo text="¿CON QUÉ DATOS SE TRABAJÓ?" color={accent} />
                      <div style={{fontFamily:"var(--font-body)",fontSize:"14px",color:"#475569"}}>
                        {results.dataset_info.nota}
                      </div>
                      <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:"10px"}}>
                        {[
                          {key:"train", emoji:"🧠", label:"ENTRENAMIENTO", color:"#38BDF8",
                           desc:"El modelo aprendió a reconocer patrones usando estos datos."},
                          {key:"val",   emoji:"🔍", label:"VALIDACIÓN",    color:"#A78BFA",
                           desc:"Datos para verificar que el modelo aprendió y no solo memorizó."},
                          {key:"test",  emoji:"✅", label:"PRUEBA FINAL",  color:"var(--c-green)",
                           desc:"Datos completamente nuevos para la evaluación final imparcial."},
                        ].map(({ key, emoji, label, color, desc }) => {
                          const d = results.dataset_info[key];
                          if (!d) return null;
                          return (
                            <div key={key} style={{background:"var(--c-dark)",border:`1px solid ${color}50`,padding:"14px"}}>
                              <div style={{fontFamily:"var(--font-px)",fontSize:"7px",color,marginBottom:"8px"}}>{emoji} {label}</div>
                              <div style={{fontFamily:"var(--font-px)",fontSize:"16px",color:"var(--c-white)",marginBottom:"4px"}}>
                                {typeof d.muestras==="number" ? d.muestras.toLocaleString() : d.muestras} muestras
                              </div>
                              <div style={{fontFamily:"var(--font-vt)",fontSize:"16px",color,marginBottom:"8px"}}>{d.porcentaje} del total</div>
                              <div style={{fontFamily:"var(--font-body)",fontSize:"13px",color:"#64748b",lineHeight:"1.4"}}>{desc}</div>
                            </div>
                          );
                        })}
                      </div>
                    </>
                  )}
                </>
              )}
            </>
          )}

          {/* ── PREDICCIONES ── */}
          {tab === "predicciones" && (
            <>
              {!results?.tabla_pred && <Vacio accent={accent} msg="Entrena o carga un modelo para ver las predicciones." />}
              {results?.tabla_pred && (
                <>
                  <Titulo text="PRIMERAS 20 PREDICCIONES VS VALORES REALES" color={accent} />
                  <p style={{fontFamily:"var(--font-body)",fontSize:"14px",color:"#64748b",lineHeight:"1.5"}}>
                    Todos los valores están en escala normalizada (0 a 1) porque así los procesa el modelo.
                    <strong style={{color:"var(--c-green)"}}> Real</strong> = lo que ocurrió de verdad.
                    <strong style={{color:accent}}> Predicción</strong> = lo que calculó el modelo.
                    El <strong style={{color:"var(--c-seismic)"}}>error</strong> entre más cerca de 0, mejor.
                  </p>
                  <table style={{width:"100%",borderCollapse:"collapse",fontFamily:"var(--font-vt)",fontSize:"17px"}}>
                    <thead>
                      <tr style={{background:`${raw}20`}}>
                        {["#","VALOR REAL","PREDICCIÓN","ERROR"].map((h,i) => (
                          <th key={i} style={{padding:"8px 12px",fontFamily:"var(--font-px)",fontSize:"7px",color:accent,textAlign:"left",borderBottom:`2px solid ${raw}50`}}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {results.tabla_pred.map((row,i) => (
                        <tr key={i} style={{background:i%2===0?"transparent":"rgba(255,255,255,.02)",borderBottom:"1px solid var(--c-border)"}}>
                          <td style={{padding:"6px 12px",color:"#475569"}}>{row.n}</td>
                          <td style={{padding:"6px 12px",color:"var(--c-green)"}}>{row.real}</td>
                          <td style={{padding:"6px 12px",color:accent}}>{row.pred}</td>
                          <td style={{padding:"6px 12px",fontFamily:"var(--font-px)",fontSize:"12px",
                            color:row.error<0.05?"var(--c-green)":row.error<0.15?"var(--c-energy)":"var(--c-seismic)"}}>
                            {row.error}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}
            </>
          )}

          {/* ── PRONÓSTICO (solo sismos) ── */}
          {tab === "pronostico" && dataset === "sismos" && (
            <>
              {!results?.pronostico?.length && <Vacio accent={accent} msg="Entrena o carga un modelo sísmico para ver el pronóstico." />}
              {results?.pronostico?.length > 0 && (() => {
                const desnorm = v => (v * (SISMOS_MAG_MAX - SISMOS_MAG_MIN) + SISMOS_MAG_MIN).toFixed(2);
                return (
                  <>
                    <Titulo text="PRONÓSTICO — PRÓXIMOS 30 DÍAS" color={accent} />
                    <p style={{fontFamily:"var(--font-body)",fontSize:"14px",color:"#64748b",lineHeight:"1.5"}}>
                      El modelo toma los últimos 14 días conocidos y estima los siguientes 30 días.
                      Los valores se muestran en <strong style={{color:"#94a3b8"}}>escala Richter real</strong> (no normalizada).
                      <br/><strong style={{color:"var(--c-energy)"}}>Esto es experimental — no reemplaza sistemas oficiales de alerta.</strong>
                    </p>
                    <table style={{width:"100%",borderCollapse:"collapse",fontFamily:"var(--font-vt)",fontSize:"17px"}}>
                      <thead>
                        <tr style={{background:`${raw}20`}}>
                          {["DÍA","MAGNITUD EST.","NIVEL","¿QUÉ SIGNIFICA?"].map((h,i) => (
                            <th key={i} style={{padding:"8px 12px",fontFamily:"var(--font-px)",fontSize:"7px",color:accent,textAlign:"left",borderBottom:`2px solid ${raw}50`}}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {results.pronostico.map((row, i) => {
                          const mag   = parseFloat(desnorm(row.valor));
                          const nivel = mag >= 6.0 ? "FUERTE" : mag >= 5.0 ? "MODERADO" : mag >= 4.0 ? "LEVE" : "MUY LEVE";
                          const nc    = nivel==="FUERTE"?"var(--c-seismic)":nivel==="MODERADO"?"var(--c-energy)":nivel==="LEVE"?"var(--c-accent)":"var(--c-green)";
                          const desc  = nivel==="FUERTE"   ? "Puede causar daños. Se siente con mucha intensidad."
                                      : nivel==="MODERADO" ? "Se siente claramente. Objetos pueden moverse."
                                      : nivel==="LEVE"     ? "Se siente. Pocas consecuencias esperadas."
                                      :                      "Apenas perceptible o no se siente.";
                          return (
                            <tr key={i} style={{background:i%2===0?"transparent":"rgba(255,255,255,.02)",borderBottom:"1px solid var(--c-border)"}}>
                              <td style={{padding:"6px 12px",color:"#475569"}}>+{row.paso} días</td>
                              <td style={{padding:"6px 12px",color:accent,fontFamily:"var(--font-px)",fontSize:"13px"}}>
                                {mag} <span style={{fontSize:"10px",color:"#475569"}}>Richter</span>
                              </td>
                              <td style={{padding:"6px 12px",color:nc,fontFamily:"var(--font-px)",fontSize:"10px"}}>{nivel}</td>
                              <td style={{padding:"6px 12px",color:"#64748b",fontSize:"15px"}}>{desc}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </>
                );
              })()}
            </>
          )}

        </div>
      </div>
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════════════════
//  COMPONENTES REUTILIZABLES
// ══════════════════════════════════════════════════════════════════════════════
function Label({ text, color }) {
  return <div style={{fontFamily:"var(--font-px)",fontSize:"7px",color,letterSpacing:"1px"}}>{text}</div>;
}
function Titulo({ text, color }) {
  return <div style={{fontFamily:"var(--font-px)",fontSize:"8px",color,letterSpacing:"1px",paddingBottom:"6px",borderBottom:`1px solid ${color}30`}}>{text}</div>;
}
function Vacio({ accent, msg }) {
  return (
    <div style={{flex:1,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",gap:"12px",padding:"40px"}}>
      <div style={{fontFamily:"var(--font-px)",fontSize:"28px",color:"var(--c-border)"}}>□</div>
      <div style={{fontFamily:"var(--font-vt)",fontSize:"18px",textAlign:"center",color:"#475569"}}>{msg}</div>
    </div>
  );
}
function Campo({ label, type, value, onChange, min, max }) {
  return (
    <div>
      <div style={{fontFamily:"var(--font-vt)",fontSize:"13px",color:"var(--c-gray)",marginBottom:"2px"}}>{label}</div>
      <input type={type} value={value} min={min} max={max} onChange={e => onChange(e.target.value)}
        style={{width:"100%",background:"#0d1220",color:"#e2e8f0",border:"1px solid var(--c-border)",fontFamily:"var(--font-vt)",fontSize:"17px",padding:"5px 8px"}} />
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════════════════
//  DECORATIVOS — PORTADA
// ══════════════════════════════════════════════════════════════════════════════
function PixelStars() {
  const stars = Array.from({length:40}, () => ({
    x: Math.random()*100, y: Math.random()*60,
    size: Math.random()>.8?2:1, delay: Math.random()*3,
  }));
  return (
    <div style={{position:"absolute",inset:0,zIndex:1}}>
      {stars.map((s,i) => (
        <div key={i} style={{position:"absolute",left:`${s.x}%`,top:`${s.y}%`,width:`${s.size*2}px`,height:`${s.size*2}px`,
          background:"#fff",animation:`blink ${1+s.delay}s steps(1) infinite`,animationDelay:`${s.delay}s`}} />
      ))}
    </div>
  );
}

function PixelCity() {
  return (
    <div style={{position:"absolute",bottom:0,left:0,right:0,height:"200px",zIndex:2}}>
      <svg viewBox="0 0 1000 200" style={{width:"100%",height:"100%",imageRendering:"pixelated"}} preserveAspectRatio="xMidYMax meet">
        {EDIFICIOS.map((b,i) => (
          <g key={i}>
            <rect x={b.x} y={200-b.h} width={b.w} height={b.h}
              fill={`rgba(13,18,32,${.8+(i%3)*.07})`} stroke="rgba(56,189,248,.12)" strokeWidth="1"/>
            {Array.from({length:Math.floor(b.h/18)}, (_,r) =>
              Array.from({length:Math.floor(b.w/12)}, (_,c) => (
                <rect key={`${r}-${c}`} x={b.x+4+c*12} y={200-b.h+6+r*18} width={6} height={8}
                  fill={Math.random()>.4?"rgba(245,158,11,.55)":"rgba(56,189,248,.25)"} />
              ))
            )}
          </g>
        ))}
      </svg>
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════════════════
//  APP PRINCIPAL
// ══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [screen,  setScreen]  = useState("portada");
  const [dataset, setDataset] = useState(null);

  useEffect(() => {
    const s = document.createElement("style");
    s.textContent = GLOBAL_CSS;
    document.head.appendChild(s);
    return () => document.head.removeChild(s);
  }, []);

  return (
    <div style={{width:"100vw",height:"100vh",overflow:"hidden"}}>
      {screen==="portada"       && <ScreenPortada  onNext={() => setScreen("seleccion")} />}
      {screen==="seleccion"     && <ScreenSeleccion onSelect={ds => { setDataset(ds); setScreen("entrenamiento"); }} />}
      {screen==="entrenamiento" && <ScreenEntrenamiento dataset={dataset} onBack={() => setScreen("seleccion")} />}
    </div>
  );
}