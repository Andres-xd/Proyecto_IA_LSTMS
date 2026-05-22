import { useState, useEffect, useRef } from "react";

const PROYECTO = {
  universidad: "UNIVERSIDAD MARIANO GÁLVEZ DE GUATEMALA",
  facultad:    "Facultad: Ingeniería en Sistemas",
  catedratico: "Catedrático: Ing. Jorge Luis Salguero Galicia",
  curso:       "Curso: INTELIGENCIA ARTIFICIAL",
  proyecto:    "Sistema de Predicción de Series de Tiempo con Redes LSTM",
  integrantes: [
    { nombre: "José Andrés López Flores",          carnet: "3590-22-6856"  },
    { nombre: "Dieter Randolfo Osorio Hernandez",  carnet: "3590-21-19249" },
    { nombre: "Jonathan Ananías Velasquez Martínez",carnet: "3590-22-2636"  },
    { nombre: "Daylin Rebeca Temaj Perez",          carnet: "3590-22-17218" },
    { nombre: "Kateryn Yohana Pasán De León",       carnet: "3590-22-9755"  },
  ],
};

/* ── Fondos pixel art ──────────────────────────────────────────
   Pon tus imágenes en frontend/public/img/
   Si no existe la imagen, se usa el fondo CSS generado.
────────────────────────────────────────────────────────────── */
const IMGS = {
  logo:       "/img/logo.png",
  bgPortada:  "/img/bg_portada.png",   // ciudad pixel art — pantalla 1
  bgEnergia:  "/img/bg_energia.png",   // fondo eléctrico — panel izq pantalla 2
  bgSismos:   "/img/bg_sismos.png",    // fondo sísmico   — panel der pantalla 2
  bgTrain:    "/img/bg_train.png",     // fondo pantalla 3
};

const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&family=Rajdhani:wght@400;600;700&display=swap');
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
  :root{
    --c-bg:#0a0e1a;--c-dark:#0d1220;--c-panel:#111827;--c-border:#2a3a5c;
    --c-energy:#F59E0B;--c-seismic:#EF4444;--c-accent:#38BDF8;
    --c-green:#4ADE80;--c-white:#F1F5F9;--c-gray:#64748B;
    --font-px:'Press Start 2P',monospace;--font-vt:'VT323',monospace;--font-body:'Rajdhani',sans-serif;
  }
  html,body,#root{width:100%;height:100%;background:var(--c-bg);color:var(--c-white);font-family:var(--font-body);overflow:hidden;}
  .px-btn{font-family:var(--font-px);font-size:10px;letter-spacing:1px;cursor:pointer;border:2px solid;padding:10px 20px;position:relative;transition:transform .05s;image-rendering:pixelated;text-transform:uppercase;background:transparent;}
  .px-btn:active{transform:translate(2px,2px);}
  .px-btn:disabled{opacity:.5;cursor:not-allowed;}
  .scanlines::before{content:'';position:absolute;inset:0;background:repeating-linear-gradient(to bottom,transparent 0px,transparent 3px,rgba(0,0,0,.08) 3px,rgba(0,0,0,.08) 4px);pointer-events:none;z-index:10;}
  .screen-enter{animation:sIn .4s steps(8,end) forwards;}
  @keyframes sIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
  .blink{animation:blink 1s steps(1) infinite;}
  @keyframes blink{50%{opacity:0}}
  .float{animation:float 3s ease-in-out infinite;}
  @keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
  .glitch{position:relative;animation:glitch 4s steps(1) infinite;}
  @keyframes glitch{0%,94%,100%{text-shadow:none}95%{text-shadow:-2px 0 var(--c-seismic),2px 0 var(--c-accent)}97%{text-shadow:2px 0 var(--c-energy),-2px 0 var(--c-green)}}
  .px-grid{background-image:linear-gradient(rgba(56,189,248,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(56,189,248,.03) 1px,transparent 1px);background-size:32px 32px;}
  ::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-track{background:var(--c-dark);}::-webkit-scrollbar-thumb{background:var(--c-border);}
  input,select{outline:none;}
`;

// ══════════════════════════════════════════════════════════════
//  PANTALLA 1 — PORTADA
// ══════════════════════════════════════════════════════════════
function ScreenPortada({ onNext }) {
  return (
    <div className="scanlines screen-enter" style={{
      width:"100vw",height:"100vh",position:"relative",overflow:"hidden",
      /* FONDO PORTADA — reemplaza con: background:`url(${IMGS.bgPortada}) center/cover no-repeat` */
      background:`radial-gradient(ellipse at 20% 80%,rgba(56,189,248,.15) 0%,transparent 50%),
                  radial-gradient(ellipse at 80% 20%,rgba(245,158,11,.10) 0%,transparent 50%),
                  linear-gradient(180deg,#050810 0%,#0a1628 50%,#0d1f3c 100%)`,
    }}>
      <PixelStars />
      <PixelCity />
      <div style={{position:"absolute",inset:0,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",gap:"14px",padding:"20px",zIndex:5}}>
        <div className="float">
          <img src={IMGS.logo} alt="Logo UMG" style={{width:"80px",height:"80px",imageRendering:"pixelated",filter:"drop-shadow(0 0 12px rgba(56,189,248,.5))"}} onError={e=>{e.target.style.display="none"}} />
        </div>
        <div style={{fontFamily:"var(--font-px)",fontSize:"8px",color:"var(--c-accent)",textAlign:"center",lineHeight:"1.8",letterSpacing:"2px",textShadow:"0 0 20px rgba(56,189,248,.6)"}}>
          {PROYECTO.universidad}
        </div>
        <div style={{fontFamily:"var(--font-vt)",fontSize:"17px",color:"#94a3b8",textAlign:"center",lineHeight:"1.5"}}>
          <div>{PROYECTO.facultad}</div>
          <div>{PROYECTO.catedratico}</div>
          <div style={{color:"var(--c-energy)",fontWeight:"bold"}}>{PROYECTO.curso}</div>
        </div>
        <div className="glitch" style={{fontFamily:"var(--font-px)",fontSize:"10px",color:"var(--c-white)",textAlign:"center",lineHeight:"2",maxWidth:"680px",textShadow:"0 0 30px rgba(255,255,255,.3)"}}>
          {PROYECTO.proyecto}
        </div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"3px 28px",background:"rgba(17,24,39,.85)",border:"1px solid var(--c-border)",padding:"10px 20px"}}>
          {PROYECTO.integrantes.map((i,idx)=>(
            <div key={idx} style={{fontFamily:"var(--font-vt)",fontSize:"16px",display:"flex",justifyContent:"space-between",gap:"20px",color:"#cbd5e1"}}>
              <span>{i.nombre}</span>
              <span style={{color:"var(--c-accent)",whiteSpace:"nowrap"}}>{i.carnet}</span>
            </div>
          ))}
        </div>
        <button className="px-btn" onClick={onNext} style={{marginTop:"10px",background:"var(--c-accent)",borderColor:"#0ea5e9",color:"#0a0e1a",fontSize:"10px",padding:"12px 32px",boxShadow:"0 0 20px rgba(56,189,248,.4)"}}>
          ▶ ¡EMPECEMOS! <span className="blink">_</span>
        </button>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
//  PANTALLA 2 — SELECCIÓN DE DATASET
// ══════════════════════════════════════════════════════════════
function ScreenSeleccion({ onSelect }) {
  const [hover,setHover]=useState(null);
  const [info,setInfo]=useState({energia:FALLBACK_INFO.energia,sismos:FALLBACK_INFO.sismos});
  useEffect(()=>{
    ["energia","sismos"].forEach(async ds=>{
      try{const r=await fetch(`http://localhost:5000/api/dataset-info/${ds}`);const d=await r.json();setInfo(p=>({...p,[ds]:d}));}catch{}
    });
  },[]);

  const datasets=[
    {key:"energia",label:"⚡ ENERGÍA ELÉCTRICA",color:"var(--c-energy)",raw:"#F59E0B",bg:IMGS.bgEnergia},
    {key:"sismos", label:"🌍 ACTIVIDAD SÍSMICA",color:"var(--c-seismic)",raw:"#EF4444",bg:IMGS.bgSismos},
  ];

  return (
    <div className="screen-enter" style={{width:"100vw",height:"100vh",display:"flex",flexDirection:"column",background:"var(--c-bg)"}}>
      <div style={{textAlign:"center",padding:"14px",borderBottom:"2px solid var(--c-border)",background:"var(--c-dark)"}}>
        <div style={{fontFamily:"var(--font-px)",fontSize:"9px",color:"var(--c-white)",letterSpacing:"2px"}}>SELECCIONA TU DATASET</div>
      </div>
      <div style={{flex:1,display:"flex",overflow:"hidden"}}>
        {datasets.map((ds,idx)=>(
          <div key={ds.key} onMouseEnter={()=>setHover(ds.key)} onMouseLeave={()=>setHover(null)}
            style={{flex:1,display:"flex",flexDirection:"column",borderRight:idx===0?"2px solid var(--c-border)":"none",
              position:"relative",overflow:"hidden",transition:"all .2s",
              /* FONDO DATASET — reemplaza con: background: hover===ds.key ? `url(${ds.bg}) center/cover` : 'var(--c-dark)' */
              background:hover===ds.key?`linear-gradient(180deg,${ds.raw}18 0%,var(--c-dark) 100%)`:"var(--c-dark)",
            }}>
            <div style={{padding:"18px 22px 10px",fontFamily:"var(--font-px)",fontSize:"11px",color:ds.color,letterSpacing:"1px",textShadow:hover===ds.key?`0 0 20px ${ds.raw}80`:"none",transition:"text-shadow .2s"}}>
              {ds.label}
            </div>
            <div style={{flex:1,padding:"0 22px",overflowY:"auto"}}>
              {info[ds.key] ? <DatasetCard info={info[ds.key]} color={ds.color} raw={ds.raw}/> : <div style={{fontFamily:"var(--font-vt)",fontSize:"18px",color:"var(--c-gray)"}}>Cargando<span className="blink">...</span></div>}
            </div>
            <div style={{padding:"16px 22px"}}>
              <button className="px-btn" onClick={()=>onSelect(ds.key)}
                style={{width:"100%",background:hover===ds.key?ds.color:"transparent",borderColor:ds.raw,color:hover===ds.key?"#0a0e1a":ds.color,fontSize:"9px",transition:"all .15s"}}>
                INICIAR ENTRENAMIENTO →
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function DatasetCard({info,color,raw}){
  return(
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
          {info.features.map((f,i)=><span key={i} style={{fontFamily:"var(--font-vt)",fontSize:"15px",background:`${raw}20`,color:"#cbd5e1",padding:"1px 8px",border:`1px solid ${raw}40`}}>{f}</span>)}
        </div>
      </div>
      <div style={{background:`${raw}15`,border:`1px solid ${raw}50`,padding:"8px 12px"}}>
        <div style={{fontFamily:"var(--font-px)",fontSize:"7px",color,marginBottom:"3px"}}>VARIABLE OBJETIVO</div>
        <div style={{fontFamily:"var(--font-vt)",fontSize:"19px",color:"#f1f5f9"}}>{info.target}</div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
//  PANTALLA 3 — ENTRENAMIENTO Y RESULTADOS (rediseñada)
// ══════════════════════════════════════════════════════════════
function ScreenEntrenamiento({dataset,onBack}){
  const isEnergia=dataset==="energia";
  const accent=isEnergia?"var(--c-energy)":"var(--c-seismic)";
  const accentRaw=isEnergia?"#F59E0B":"#EF4444";

  const [config,setConfig]=useState({
    epochs:isEnergia?10:50,
    batch_size:64,
    model_name:`model_${dataset}_custom`,
    agent_type:"lstm_simple"
  });
  const [progress,setProgress]=useState(null);
  const [results,setResults]=useState(null);
  const [polling,setPolling]=useState(false);
  const [models,setModels]=useState([]);
  const [loadName,setLoadName]=useState("");
  const [tab,setTab]=useState("entrenamiento"); // "entrenamiento" | "datos" | "predicciones" | "forecast"
  const logRef=useRef(null);
  const intervalRef=useRef(null);

  useEffect(()=>{
    fetch("http://localhost:5000/api/models")
      .then(r=>r.json())
      .then(data=>{
        console.log("Modelos recibidos:", data);
        setModels(data);
      })
      .catch(err=>console.error("Error al cargar modelos:", err));
  },[]);

  // Auto-scroll del log
  useEffect(()=>{
    if(logRef.current) logRef.current.scrollTop=logRef.current.scrollHeight;
  },[progress?.epoch_log]);

  useEffect(()=>{
    if(polling){
      intervalRef.current=setInterval(async()=>{
        try{
          const r=await fetch("http://localhost:5000/api/train/progress");
          const d=await r.json();
          setProgress(d);
          if(d.done){setPolling(false);clearInterval(intervalRef.current);if(!d.error){setResults(d);setTab("datos");}}
        }catch{}
      },800);
    }
    return()=>clearInterval(intervalRef.current);
  },[polling]);

  async function handleTrain(){
    setResults(null);setProgress(null);setTab("entrenamiento");
    try{
      await fetch("http://localhost:5000/api/train",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({dataset,...config})});
      setPolling(true);
    }catch{alert("No se pudo conectar con Flask en localhost:5000");}
  }

  async function handleLoad(){
    if(!loadName)return;
    try{
      const r=await fetch("http://localhost:5000/api/load-model",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({dataset,model_name:loadName})});
      const d=await r.json();
      if(d.error){alert(d.error);return;}
      setResults(d);setProgress({epoch_log:d.epoch_log,done:true,running:false});setTab("datos");
    }catch{alert("Error al cargar el modelo");}
  }

  const pct=progress?Math.round((progress.epoch/(progress.total_epochs||1))*100):0;
  const isRunning=progress?.running;

  const tabs=["entrenamiento","datos","predicciones"];
  if(dataset==="sismos") tabs.push("forecast");

  return(
    <div className="screen-enter px-grid" style={{
      width:"100vw",height:"100vh",display:"flex",overflow:"hidden",
      /* FONDO PANTALLA 3 — reemplaza con: background:`url(${IMGS.bgTrain}) center/cover,var(--c-bg)` */
      background:"var(--c-bg)",
    }}>

      {/* ── PANEL IZQUIERDO ── */}
      <div style={{width:"280px",minWidth:"280px",background:"var(--c-dark)",borderRight:`2px solid ${accentRaw}50`,display:"flex",flexDirection:"column",overflowY:"auto"}}>
        <div style={{padding:"14px",borderBottom:`2px solid ${accentRaw}30`,background:`${accentRaw}12`}}>
          <button onClick={onBack} style={{fontFamily:"var(--font-px)",fontSize:"7px",background:"transparent",border:"none",color:"var(--c-gray)",cursor:"pointer",marginBottom:"6px"}}>← VOLVER</button>
          <div style={{fontFamily:"var(--font-px)",fontSize:"8px",color:accent,lineHeight:"1.8"}}>
            {isEnergia?"⚡ ENERGÍA ELÉCTRICA":"🌍 ACTIVIDAD SÍSMICA"}
          </div>
          {isEnergia&&<div style={{fontFamily:"var(--font-vt)",fontSize:"13px",color:"#475569",marginTop:"4px"}}>Prototipo: 25,000 de 1.4M muestras</div>}
        </div>

        <div style={{padding:"14px",display:"flex",flexDirection:"column",gap:"14px"}}>
          {/* Selector de Arquitectura */}
          <SideSection title="ARQUITECTURA DEL AGENTE" color={accent}>
            <select 
              value={config.agent_type || "lstm_simple"} 
              onChange={e=>setConfig(p=>({...p,agent_type:e.target.value}))}
              style={{width:"100%",background:"#0d1220",color:"#e2e8f0",border:`1px solid ${accentRaw}50`,fontFamily:"var(--font-vt)",fontSize:"16px",padding:"5px"}}>
              <option value="lstm_simple">LSTM Simple (Baseline)</option>
              <option value="stacked_lstm">Stacked LSTM (Multicapa)</option>
              <option value="bilstm">Bidirectional LSTM</option>
              <option value="cnn_lstm">CNN-LSTM Híbrido</option>
            </select>
            <div style={{fontFamily:"var(--font-vt)",fontSize:"13px",color:"#64748b",marginTop:"6px",lineHeight:"1.4"}}>
              {config.agent_type === "lstm_simple" && "Arquitectura básica de 2 capas LSTM"}
              {config.agent_type === "stacked_lstm" && "Múltiples capas LSTM apiladas"}
              {config.agent_type === "bilstm" && "Lee la secuencia hacia adelante y atrás"}
              {config.agent_type === "cnn_lstm" && "CNN extrae patrones locales + LSTM temporal"}
            </div>
          </SideSection>

          {/* Parámetros */}
          <SideSection title="PARÁMETROS" color={accent}>
            <CField label="ÉPOCAS" type="number" min={1} max={100} value={config.epochs} onChange={v=>setConfig(p=>({...p,epochs:+v}))}/>
            <CField label="BATCH SIZE" type="number" value={config.batch_size} onChange={v=>setConfig(p=>({...p,batch_size:+v}))}/>
            <CField label="NOMBRE DEL MODELO (.pth)" type="text" value={config.model_name} onChange={v=>setConfig(p=>({...p,model_name:v}))}/>
          </SideSection>

          <button className="px-btn" onClick={handleTrain} disabled={isRunning}
            style={{background:isRunning?accentRaw:accent,borderColor:accentRaw,color:"#0a0e1a",width:"100%",fontSize:"8px"}}>
            {isRunning?"⟳ ENTRENANDO...":"▶ ENTRENAR MODELO"}
          </button>

          {/* Cargar modelo */}
          <SideSection title="CARGAR MODELO EXISTENTE" color={accent}>
            <select value={loadName} onChange={e=>setLoadName(e.target.value)}
              style={{width:"100%",background:"#0d1220",color:"#e2e8f0",border:`1px solid ${accentRaw}50`,fontFamily:"var(--font-vt)",fontSize:"16px",padding:"5px"}}>
              <option value="">-- selecciona --</option>
              {models.filter(m=>m.dataset===dataset||m.name.includes(dataset)).length === 0 ? (
                <option disabled>No hay modelos entrenados aún</option>
              ) : (
                models.filter(m=>m.dataset===dataset||m.name.includes(dataset)).map(m=><option key={m.name} value={m.name}>{m.display_name||m.name} ({m.size_kb}KB)</option>)
              )}
            </select>
            <button className="px-btn" onClick={handleLoad} disabled={!loadName}
              style={{marginTop:"6px",width:"100%",borderColor:accentRaw,color:accent,fontSize:"8px"}}>
              CARGAR Y EVALUAR
            </button>
            <div style={{fontFamily:"var(--font-vt)",fontSize:"14px",color:"var(--c-gray)",marginTop:"6px"}}>
              {models.filter(m=>m.dataset===dataset||m.name.includes(dataset)).length} modelo(s) disponible(s)
            </div>
          </SideSection>
        </div>
      </div>

      {/* ── PANEL DERECHO ── */}
      <div style={{flex:1,display:"flex",flexDirection:"column",overflow:"hidden"}}>

        {/* Tabs */}
        <div style={{display:"flex",borderBottom:`2px solid var(--c-border)`,background:"var(--c-dark)"}}>
          {tabs.map(t=>(
            <button key={t} onClick={()=>setTab(t)}
              style={{fontFamily:"var(--font-px)",fontSize:"7px",padding:"12px 16px",border:"none",cursor:"pointer",letterSpacing:"1px",textTransform:"uppercase",
                background:tab===t?`${accentRaw}20`:"transparent",
                color:tab===t?accent:"var(--c-gray)",
                borderBottom:tab===t?`2px solid ${accentRaw}`:"2px solid transparent",
              }}>
              {t==="entrenamiento"?"📟 PROCESO":t==="datos"?"📊 RESULTADOS":t==="predicciones"?"🔢 PREDICCIONES":"📅 PRONÓSTICO"}
            </button>
          ))}
        </div>

        {/* Contenido tab */}
        <div style={{flex:1,overflowY:"auto",padding:"18px",display:"flex",flexDirection:"column",gap:"14px"}}>

          {/* TAB: PROCESO DE ENTRENAMIENTO */}
          {tab==="entrenamiento"&&(
            <>
              {/* Barra de progreso */}
              {progress&&(
                <div style={{background:"var(--c-dark)",border:`1px solid ${accentRaw}40`,padding:"14px"}}>
                  <div style={{display:"flex",justifyContent:"space-between",marginBottom:"8px"}}>
                    <span style={{fontFamily:"var(--font-px)",fontSize:"8px",color:accent}}>
                      {isRunning?`ÉPOCA ${progress.epoch} / ${progress.total_epochs}`:"COMPLETADO"}
                    </span>
                    <span style={{fontFamily:"var(--font-px)",fontSize:"8px",color:"var(--c-green)"}}>{pct}%</span>
                  </div>
                  {/* Barra */}
                  <div style={{width:"100%",height:"18px",background:"#1e293b",border:"2px solid #334155",position:"relative",overflow:"hidden"}}>
                    <div style={{height:"100%",width:`${pct}%`,background:`linear-gradient(90deg,${accentRaw},${accentRaw}aa)`,transition:"width .3s steps(20,end)",imageRendering:"pixelated"}}/>
                    {/* Efecto pixel */}
                    <div style={{position:"absolute",inset:0,backgroundImage:`repeating-linear-gradient(90deg,transparent,transparent 7px,rgba(0,0,0,.15) 7px,rgba(0,0,0,.15) 8px)`,pointerEvents:"none"}}/>
                  </div>
                  {isRunning&&(
                    <div style={{display:"flex",gap:"20px",marginTop:"8px",fontFamily:"var(--font-vt)",fontSize:"17px"}}>
                      <span>Train Loss: <span style={{color:"var(--c-green)"}}>{progress.train_loss}</span></span>
                      <span>Val Loss: <span style={{color:accent}}>{progress.val_loss}</span></span>
                    </div>
                  )}
                </div>
              )}

              {/* Log de épocas en tiempo real */}
              <div style={{background:"#050810",border:`1px solid ${accentRaw}30`,flex:1,minHeight:"300px",display:"flex",flexDirection:"column"}}>
                <div style={{padding:"8px 12px",borderBottom:`1px solid ${accentRaw}20`,fontFamily:"var(--font-px)",fontSize:"7px",color:accent}}>
                  CONSOLA — LOG EN TIEMPO REAL
                </div>
                <div ref={logRef} style={{flex:1,overflowY:"auto",padding:"10px 14px",fontFamily:"var(--font-vt)",fontSize:"16px",lineHeight:"1.6",color:"#94a3b8",maxHeight:"420px"}}>
                  {progress?.epoch_log?.length>0
                    ? progress.epoch_log.map((l,i)=>(
                        <div key={i} style={{color:l.includes("MEJORA")?"var(--c-green)":l.includes("ERROR")?"var(--c-seismic)":l.includes("✅")?"var(--c-green)":l.includes("⏹")?"var(--c-energy)":"#64748b",whiteSpace:"pre"}}>
                          {l}
                        </div>
                      ))
                    : <div style={{color:"#334155"}}>Presiona "ENTRENAR MODELO" para iniciar<span className="blink">_</span></div>
                  }
                </div>
              </div>
            </>
          )}

          {/* TAB: RESULTADOS */}
          {tab==="datos"&&(
            <>
              {!results&&<EmptyState accent={accent} msg="Entrena o carga un modelo para ver los resultados." />}

              {results?.metrics&&(
                <>
                  {/* Métricas con explicación */}
                  <SectionTitle title="¿QUÉ TAN BIEN PREDICE EL MODELO?" color={accent}/>
                  <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:"10px"}}>
                    {[
                      {label:"RMSE",val:results.metrics.rmse,
                        que:"Error cuadrático medio",
                        explica:"Mide cuánto se equivoca el modelo en promedio. Penaliza más los errores grandes. Cuanto más cercano a 0, mejor.",
                        bueno:"< 0.05 es excelente",color:"var(--c-accent)"},
                      {label:"MAE",val:results.metrics.mae,
                        que:"Error absoluto medio",
                        explica:"El error promedio de cada predicción individual. Más fácil de interpretar que el RMSE. También, más cercano a 0 es mejor.",
                        bueno:"< 0.03 es excelente",color:"var(--c-green)"},
                      {label:"MAPE",val:`${results.metrics.mape}%`,
                        que:"Error porcentual medio",
                        explica:"El error expresado como porcentaje del valor real. Por ejemplo, 7.81% significa que el modelo se equivoca en promedio un 7.81%.",
                        bueno:"< 10% es aceptable",color:accent},
                    ].map((m,i)=>(
                      <div key={i} style={{background:"var(--c-dark)",border:`2px solid ${accentRaw}50`,padding:"14px"}}>
                        <div style={{fontFamily:"var(--font-px)",fontSize:"7px",color:m.color,marginBottom:"6px"}}>{m.label}</div>
                        <div style={{fontFamily:"var(--font-px)",fontSize:"18px",color:"var(--c-white)",marginBottom:"6px"}}>{m.val}</div>
                        <div style={{fontFamily:"var(--font-vt)",fontSize:"15px",color:m.color,marginBottom:"4px"}}>{m.que}</div>
                        <div style={{fontFamily:"var(--font-body)",fontSize:"13px",color:"#64748b",lineHeight:"1.4",marginBottom:"4px"}}>{m.explica}</div>
                        <div style={{fontFamily:"var(--font-vt)",fontSize:"14px",color:"var(--c-green)"}}>{m.bueno}</div>
                      </div>
                    ))}
                  </div>

                  {/* División de datos */}
                  {results.dataset_info&&(
                    <>
                      <SectionTitle title="¿CON QUÉ DATOS SE TRABAJÓ?" color={accent}/>
                      <div style={{fontFamily:"var(--font-body)",fontSize:"14px",color:"#475569",marginTop:"-6px"}}>
                        {results.dataset_info.nota}
                      </div>
                      <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:"10px"}}>
                        {[
                          {key:"train",emoji:"🧠",color:"#38BDF8"},
                          {key:"val",  emoji:"🔍",color:"#A78BFA"},
                          {key:"test", emoji:"✅",color:"var(--c-green)"},
                        ].map(({key,emoji,color})=>{
                          const d=results.dataset_info[key];
                          if(!d)return null;
                          return(
                            <div key={key} style={{background:"var(--c-dark)",border:`1px solid ${color}50`,padding:"14px"}}>
                              <div style={{fontFamily:"var(--font-px)",fontSize:"7px",color,marginBottom:"8px"}}>{emoji} {d.uso.toUpperCase()}</div>
                              <div style={{fontFamily:"var(--font-px)",fontSize:"14px",color:"var(--c-white)",marginBottom:"4px"}}>{d.muestras.toLocaleString()}</div>
                              <div style={{fontFamily:"var(--font-vt)",fontSize:"16px",color,marginBottom:"6px"}}>{d.porcentaje} del total</div>
                              <div style={{fontFamily:"var(--font-body)",fontSize:"13px",color:"#64748b",lineHeight:"1.4"}}>{d.descripcion}</div>
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

          {/* TAB: PREDICCIONES */}
          {tab==="predicciones"&&(
            <>
              {!results?.pred_table&&<EmptyState accent={accent} msg="Entrena o carga un modelo para ver las predicciones." />}
              {results?.pred_table&&(
                <>
                  <SectionTitle title="PRIMERAS 20 PREDICCIONES VS VALORES REALES" color={accent}/>
                  <div style={{fontFamily:"var(--font-body)",fontSize:"14px",color:"#64748b",marginTop:"-6px",lineHeight:"1.5"}}>
                    Estos son valores <strong style={{color:"#94a3b8"}}>normalizados</strong> (escala 0 a 1).
                    La columna <strong style={{color:"var(--c-green)"}}>Real</strong> es lo que realmente ocurrió.
                    La columna <strong style={{color:accent}}>Predicción</strong> es lo que el modelo calculó.
                    El <strong style={{color:"var(--c-seismic)"}}>Error</strong> es la diferencia entre ambos — entre más cercano a 0, mejor.
                  </div>
                  <table style={{width:"100%",borderCollapse:"collapse",fontFamily:"var(--font-vt)",fontSize:"17px"}}>
                    <thead>
                      <tr style={{background:`${accentRaw}20`}}>
                        {["#","VALOR REAL","PREDICCIÓN","ERROR ABSOLUTO"].map((h,i)=>(
                          <th key={i} style={{padding:"8px 12px",fontFamily:"var(--font-px)",fontSize:"7px",color:accent,textAlign:"left",borderBottom:`2px solid ${accentRaw}50`}}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {results.pred_table.map((row,i)=>(
                        <tr key={i} style={{background:i%2===0?"transparent":"rgba(255,255,255,.02)",borderBottom:"1px solid var(--c-border)"}}>
                          <td style={{padding:"6px 12px",color:"#475569"}}>{row.n}</td>
                          <td style={{padding:"6px 12px",color:"var(--c-green)"}}>{row.real}</td>
                          <td style={{padding:"6px 12px",color:accent}}>{row.pred}</td>
                          <td style={{padding:"6px 12px",color:row.error<0.05?"var(--c-green)":row.error<0.15?"var(--c-energy)":"var(--c-seismic)",fontFamily:"var(--font-px)",fontSize:"12px"}}>{row.error}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}
            </>
          )}

          {/* TAB: FORECAST (solo sismología) */}
          {tab==="forecast"&&dataset==="sismos"&&(
            <>
              {!results?.forecast?.length&&<EmptyState accent={accent} msg="Entrena o carga un modelo sísmico para ver el pronóstico." />}
              {results?.forecast?.length>0&&(
                <>
                  <SectionTitle title="PRONÓSTICO — PRÓXIMOS 30 DÍAS" color={accent}/>
                  <div style={{fontFamily:"var(--font-body)",fontSize:"14px",color:"#64748b",marginTop:"-6px",lineHeight:"1.5"}}>
                    El modelo usa los últimos 14 días conocidos y predice los siguientes 30 días de forma iterativa.
                    Los valores son la <strong style={{color:"#94a3b8"}}>magnitud promedio diaria normalizada</strong> (escala 0–1).
                    Valores cercanos a 0 = baja actividad. Valores cercanos a 1 = alta actividad sísmica.
                    <br/><strong style={{color:"var(--c-energy)"}}>⚠ Este pronóstico es experimental, no un sistema de alerta oficial.</strong>
                  </div>
                  <table style={{width:"100%",borderCollapse:"collapse",fontFamily:"var(--font-vt)",fontSize:"17px"}}>
                    <thead>
                      <tr style={{background:`${accentRaw}20`}}>
                        {["DÍA","MAGNITUD PREDICHA (norm.)","NIVEL ESTIMADO"].map((h,i)=>(
                          <th key={i} style={{padding:"8px 12px",fontFamily:"var(--font-px)",fontSize:"7px",color:accent,textAlign:"left",borderBottom:`2px solid ${accentRaw}50`}}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {results.forecast.map((row,i)=>{
                        const nivel=row.valor>0.4?"ALTO":row.valor>0.15?"MEDIO":"BAJO";
                        const nc=nivel==="ALTO"?"var(--c-seismic)":nivel==="MEDIO"?"var(--c-energy)":"var(--c-green)";
                        return(
                          <tr key={i} style={{background:i%2===0?"transparent":"rgba(255,255,255,.02)",borderBottom:"1px solid var(--c-border)"}}>
                            <td style={{padding:"6px 12px",color:"#64748b"}}>+{row.dia} días</td>
                            <td style={{padding:"6px 12px",color:accent}}>{row.valor}</td>
                            <td style={{padding:"6px 12px",color:nc,fontFamily:"var(--font-px)",fontSize:"11px"}}>{nivel}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Componentes pequeños ──────────────────────────────────────
function SideSection({title,color,children}){
  return(
    <div>
      <div style={{fontFamily:"var(--font-px)",fontSize:"7px",color,marginBottom:"8px",letterSpacing:"1px"}}>{title}</div>
      <div style={{display:"flex",flexDirection:"column",gap:"7px"}}>{children}</div>
    </div>
  );
}
function CField({label,type,value,onChange,min,max}){
  return(
    <div>
      <div style={{fontFamily:"var(--font-vt)",fontSize:"13px",color:"var(--c-gray)",marginBottom:"2px"}}>{label}</div>
      <input type={type} value={value} min={min} max={max} onChange={e=>onChange(e.target.value)}
        style={{width:"100%",background:"#0d1220",color:"#e2e8f0",border:"1px solid var(--c-border)",fontFamily:"var(--font-vt)",fontSize:"17px",padding:"5px 8px"}}/>
    </div>
  );
}
function SectionTitle({title,color}){
  return <div style={{fontFamily:"var(--font-px)",fontSize:"8px",color,letterSpacing:"1px",paddingBottom:"6px",borderBottom:`1px solid ${color}30`}}>{title}</div>;
}
function EmptyState({accent,msg}){
  return(
    <div style={{flex:1,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",gap:"12px",color:"var(--c-gray)",padding:"40px"}}>
      <div style={{fontFamily:"var(--font-px)",fontSize:"28px"}}>□</div>
      <div style={{fontFamily:"var(--font-vt)",fontSize:"18px",textAlign:"center",color:"#475569"}}>{msg}</div>
    </div>
  );
}

// ── Decorativos ───────────────────────────────────────────────
function PixelStars(){
  const stars=Array.from({length:40},()=>({x:Math.random()*100,y:Math.random()*60,size:Math.random()>.8?2:1,delay:Math.random()*3}));
  return(
    <div style={{position:"absolute",inset:0,zIndex:1}}>
      {stars.map((s,i)=><div key={i} style={{position:"absolute",left:`${s.x}%`,top:`${s.y}%`,width:`${s.size*2}px`,height:`${s.size*2}px`,background:"#fff",imageRendering:"pixelated",animation:`blink ${1+s.delay}s steps(1) infinite`,animationDelay:`${s.delay}s`}}/>)}
    </div>
  );
}
function PixelCity(){
  const buildings=[{x:0,w:60,h:80},{x:55,w:40,h:120},{x:90,w:80,h:60},{x:165,w:50,h:100},{x:210,w:120,h:80},{x:325,w:60,h:140},{x:380,w:90,h:90},{x:465,w:50,h:110},{x:510,w:70,h:70},{x:575,w:100,h:120},{x:670,w:60,h:90},{x:725,w:80,h:60},{x:800,w:50,h:130},{x:845,w:90,h:80},{x:930,w:60,h:100}];
  return(
    <div style={{position:"absolute",bottom:0,left:0,right:0,height:"200px",zIndex:2}}>
      <svg viewBox="0 0 1000 200" style={{width:"100%",height:"100%",imageRendering:"pixelated"}} preserveAspectRatio="xMidYMax meet">
        {buildings.map((b,i)=>(
          <g key={i}>
            <rect x={b.x} y={200-b.h} width={b.w} height={b.h} fill={`rgba(13,18,32,${.8+(i%3)*.07})`} stroke="rgba(56,189,248,.12)" strokeWidth="1"/>
            {Array.from({length:Math.floor(b.h/18)},(_,r)=>Array.from({length:Math.floor(b.w/12)},(_,c)=>(
              <rect key={`${r}-${c}`} x={b.x+4+c*12} y={200-b.h+6+r*18} width={6} height={8} fill={Math.random()>.4?"rgba(245,158,11,.55)":"rgba(56,189,248,.25)"}/>
            )))}
          </g>
        ))}
      </svg>
    </div>
  );
}

const FALLBACK_INFO={
  energia:{nombre:"Consumo Eléctrico",fuente:"UCI ML Repository",registros:"2,075,259",periodo:"2006 — 2010",frecuencia:"1 medición/minuto",features:["Global_active_power","Global_reactive_power","Voltage","Global_intensity","Sub_metering_1","Sub_metering_2","Sub_metering_3"],target:"Global_active_power",descripcion:"Mediciones de consumo eléctrico de un hogar en Francia. Permite predecir la demanda energética para optimizar redes eléctricas inteligentes.",window_size:60,color:"#F59E0B"},
  sismos:{nombre:"Actividad Sísmica",fuente:"USGS Earthquake Catalog",registros:"3,586 eventos",periodo:"1970 — 2026",frecuencia:"Serie diaria agregada",features:["avg_mag","max_mag","avg_depth","count_sismos","hay_actividad"],target:"avg_mag (magnitud promedio diaria)",descripcion:"Registro sísmico en Centroamérica. Útil para analizar patrones en series sísmicas para sistemas de monitoreo en Guatemala.",window_size:14,color:"#EF4444"},
};

// ══════════════════════════════════════════════════════════════
//  APP PRINCIPAL
// ══════════════════════════════════════════════════════════════
export default function App(){
  const [screen,setScreen]=useState("portada");
  const [dataset,setDataset]=useState(null);

  useEffect(()=>{
    const s=document.createElement("style");
    s.textContent=GLOBAL_CSS;
    document.head.appendChild(s);
    return()=>document.head.removeChild(s);
  },[]);

  return(
    <div style={{width:"100vw",height:"100vh",overflow:"hidden"}}>
      {screen==="portada"&&<ScreenPortada onNext={()=>setScreen("seleccion")}/>}
      {screen==="seleccion"&&<ScreenSeleccion onSelect={ds=>{setDataset(ds);setScreen("entrenamiento");}}/>}
      {screen==="entrenamiento"&&<ScreenEntrenamiento dataset={dataset} onBack={()=>setScreen("seleccion")}/>}
    </div>
  );
}