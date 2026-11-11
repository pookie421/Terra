import {useEffect,useRef,useState,type MutableRefObject} from 'react';
import * as maplibregl from 'maplibre-gl';
import type {ExpressionSpecification,GeoJSONSource,StyleSpecification} from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url';
import {AlertTriangle,Compass,Maximize} from 'lucide-react';
import type {FeatureCollection,Geometry,Position} from 'geojson';
import type {Scan,Cell,Runout} from '../types';

maplibregl.setWorkerUrl(workerUrl);
const EMPTY:FeatureCollection={type:'FeatureCollection',features:[]};
export type MapLayer='score'|'fs'|'slope'|'pressure';
export interface MapHandle {
  flyTo:(cell:Cell)=>void;
  scanCenter:()=>void;
  fitScan:()=>void;
}
interface Props {
  scan:Scan|null;
  selected:string;
  onSelect:(id:string)=>void;
  onScanCoordinate:(latitude:number,longitude:number)=>void;
  handle:MutableRefObject<MapHandle|null>;
  runout?:Runout;
  layer:MapLayer;
  onCenterChange?:(latitude:number,longitude:number)=>void;
  busy?:boolean;
}

const LAYER_META:Record<MapLayer,{label:string;left:string;right:string;gradient:string}>={
  score:{label:'Initiation score',left:'0',right:'1',gradient:'linear-gradient(90deg,#699886,#b9b285,#ce9b66,#bd6966)'},
  fs:{label:'Factor of safety',left:'≤ 0.5',right:'≥ 3',gradient:'linear-gradient(90deg,#bd6966,#ce9b66,#9ca783,#699886)'},
  slope:{label:'Slope',left:'0°',right:'60°+',gradient:'linear-gradient(90deg,#7392a6,#809e90,#c1a574,#bc7770)'},
  pressure:{label:'Pore-pressure head',left:'0 m',right:'3 m+',gradient:'linear-gradient(90deg,#7392a6,#809e90,#c1a574,#bc7770)'},
};

function longitude(value:number):number{return ((value+180)%360+360)%360-180;}
function valueFor(layer:MapLayer):ExpressionSpecification {
  if(layer==='score')return ['coalesce',['get','score'],['get','physics_informed_score'],-1];
  return ['coalesce',['get',layer==='fs'?'fs_mid':layer==='slope'?'slope_deg':'pressure_head_m'],-1];
}
function colorFor(layer:MapLayer):ExpressionSpecification {
  const value=valueFor(layer);
  const ramp:ExpressionSpecification=layer==='fs'
    ?['interpolate',['linear'],value,0,'#bd6966',1,'#ce9b66',1.5,'#9ca783',3,'#699886']
    :layer==='slope'
      ?['interpolate',['linear'],value,0,'#7392a6',15,'#809e90',35,'#c1a574',60,'#bc7770']
      :layer==='pressure'
        ?['interpolate',['linear'],value,0,'#7392a6',.5,'#809e90',1.5,'#c1a574',3,'#bc7770']
        :['interpolate',['linear'],value,0,'#699886',.3,'#b9b285',.65,'#ce9b66',1,'#bd6966'];
  // Missing evidence has its own neutral color. It is never rendered as zero hazard.
  return ['case',['<',value,0],'#737c8a',ramp];
}
function coordinates(geometry:Geometry):Position[] {
  switch(geometry.type){
    case 'Point':return [geometry.coordinates];
    case 'MultiPoint':case 'LineString':return geometry.coordinates;
    case 'MultiLineString':case 'Polygon':return geometry.coordinates.flat();
    case 'MultiPolygon':return geometry.coordinates.flat(2);
    case 'GeometryCollection':return geometry.geometries.flatMap(coordinates);
  }
}
function localGeometry(geometry:Geometry,anchor:number):Geometry {
  const position=(p:Position):Position=>[anchor+longitude(p[0]-anchor),...p.slice(1)];
  // Adjacent cells/routes can straddle ±180°. Preserve their short local edges
  // instead of drawing a polygon or route across almost the whole planet.
  switch(geometry.type){
    case 'Point':return {...geometry,coordinates:position(geometry.coordinates)};
    case 'MultiPoint':case 'LineString':return {...geometry,coordinates:geometry.coordinates.map(position)};
    case 'MultiLineString':case 'Polygon':return {...geometry,coordinates:geometry.coordinates.map(r=>r.map(position))};
    case 'MultiPolygon':return {...geometry,coordinates:geometry.coordinates.map(p=>p.map(r=>r.map(position)))};
    case 'GeometryCollection':return {...geometry,geometries:geometry.geometries.map(g=>localGeometry(g,anchor))};
  }
}

export default function MapCanvas({scan,selected,onSelect,onScanCoordinate,handle,runout,layer,onCenterChange}:Props){
  const host=useRef<HTMLDivElement>(null),map=useRef<maplibregl.Map|null>(null),reticle=useRef<maplibregl.Marker|null>(null);
  const callbacks=useRef({onSelect,onScanCoordinate,onCenterChange});
  const current=useRef({scan,selected,runout,layer});
  const previousScan=useRef<string|null>(null);
  const [error,setError]=useState(''),[ready,setReady]=useState(false);
  callbacks.current={onSelect,onScanCoordinate,onCenterChange};
  current.current={scan,selected,runout,layer};

  function markTarget(latitude:number,lon:number){
    const m=map.current;if(!m)return;
    if(!reticle.current){
      const target=document.createElement('div');target.className='scan-marker';
      target.setAttribute('aria-label','Scan location');
      target.style.cssText='width:12px;height:12px;border:2px solid #fff;border-radius:50%;box-shadow:0 1px 3px rgba(0,0,0,.3);pointer-events:none;background:#6d879e;';
      reticle.current=new maplibregl.Marker({element:target,anchor:'center'}).setLngLat([lon,latitude]).addTo(m);
    }else reticle.current.setLngLat([lon,latitude]);
  }
  function scanAt(latitude:number,lon:number){
    const normalized=longitude(lon);markTarget(latitude,normalized);
    callbacks.current.onScanCoordinate(latitude,normalized);
  }
  function fitCurrent(){
    const m=map.current,s=current.current.scan;if(!m||!s)return;
    const anchor=s.location.longitude;
    const positions=s.h3_geojson.features.flatMap(f=>coordinates(f.geometry));
    if(!positions.length){m.flyTo({center:[s.location.longitude,s.location.latitude],zoom:12,pitch:55});return;}
    const bounds=new maplibregl.LngLatBounds();
    for(const [lon,lat] of positions){
      // Keep a dateline-crossing scan local instead of fitting almost the whole world.
      bounds.extend([anchor+longitude(lon-anchor),lat]);
    }
    const width=m.getContainer().clientWidth;
    const desktop=width>1150;
    m.fitBounds(bounds,{padding:desktop?{top:150,bottom:80,left:360,right:400}:{top:150,bottom:110,left:40,right:40},maxZoom:14.6,pitch:55,bearing:-18,duration:1400,essential:false});
    markTarget(s.location.latitude,s.location.longitude);
  }
  function updateLayers(){
    const m=map.current,{scan:s,selected:id,runout:r,layer:active}=current.current;
    if(!m||!m.getSource('cells'))return;
    const physical=new Map(s?.physical_assessment.cells.map(c=>[c.h3_index,c]));
    const learned=new Map(s?.physics_informed_assessment?.cells.map(c=>[c.h3_index,c]));
    const data:FeatureCollection<Geometry,Cell>=s?{...s.h3_geojson,features:s.h3_geojson.features.map(f=>({...f,geometry:localGeometry(f.geometry,s.location.longitude),properties:{...physical.get(f.properties.h3_index),...f.properties,...learned.get(f.properties.h3_index)}}))}:{type:'FeatureCollection',features:[]};
    (m.getSource('cells') as GeoJSONSource).setData(data);
    (m.getSource('selected') as GeoJSONSource).setData({...data,features:data.features.filter(f=>f.properties.h3_index===id)});
    const selectedRoute=r?.h3_index===id?r:undefined;
    const routes=selectedRoute?.scenarios?.filter(v=>(v.coordinates_lonlat?.length??0)>1).map((v,i)=>({type:'Feature' as const,geometry:localGeometry({type:'LineString',coordinates:v.coordinates_lonlat!},s?.location.longitude??0),properties:{scenario:i}}));
    const sectorRoutes=s?{...s.runout_geojson,features:s.runout_geojson.features.map(f=>({...f,geometry:localGeometry(f.geometry,s.location.longitude)}))}:EMPTY;
    (m.getSource('runout') as GeoJSONSource).setData(routes?{type:'FeatureCollection',features:routes}:sectorRoutes);
    const corridor=selectedRoute?.corridor_h3_indices;
    (m.getSource('corridor') as GeoJSONSource).setData(corridor?{...data,features:data.features.filter(f=>corridor.includes(f.properties.h3_index))}:s?.runout_corridor_geojson??EMPTY);
    m.setPaintProperty('hexagons','fill-extrusion-color',colorFor(active));
  }

  useEffect(()=>{
    if(!host.current)return;
    const style:StyleSpecification={
      version:8,
      projection:{type:'globe'},
      sources:{
        imagery:{type:'raster',tiles:['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],tileSize:256,maxzoom:19,attribution:'Imagery © Esri, Maxar, Earthstar Geographics and contributors'},
        terrain:{type:'raster-dem',tiles:['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],tileSize:256,encoding:'terrarium',maxzoom:15,attribution:'Elevation © Mapzen / upstream DEM contributors'},
        cells:{type:'geojson',data:EMPTY},selected:{type:'geojson',data:EMPTY},runout:{type:'geojson',data:EMPTY},corridor:{type:'geojson',data:EMPTY},
      },
      terrain:{source:'terrain',exaggeration:1.35},
      sky:{'sky-color':'#14181d','horizon-color':'#30383e','fog-color':'#14181d','sky-horizon-blend':.7,'horizon-fog-blend':.7,'fog-ground-blend':.8},
      layers:[
        {id:'background',type:'background',paint:{'background-color':'#14181d'}},
        {id:'satellite',type:'raster',source:'imagery',paint:{'raster-saturation':-.12,'raster-brightness-max':.92,'raster-contrast':.02}},
        {id:'corridor-fill',type:'fill',source:'corridor',paint:{'fill-color':'#c6a16b','fill-opacity':.16}},
        {id:'hexagons',type:'fill-extrusion',source:'cells',paint:{'fill-extrusion-color':colorFor(current.current.layer),'fill-extrusion-height':['+',12,['*',180,['max',0,valueFor('score')]]],'fill-extrusion-base':0,'fill-extrusion-opacity':.58}},
        {id:'hex-edges',type:'line',source:'cells',paint:{'line-color':'#b4bfc6','line-width':.6,'line-opacity':.3}},
        {id:'runout-lines',type:'line',source:'runout',paint:{'line-color':['match',['get','scenario'],0,'#d5b47e',1,'#c99a70',2,'#e4cd9c','#c6a16b'],'line-width':2.5,'line-opacity':.95,'line-dasharray':[2,1]}},
        {id:'selected-outline',type:'line',source:'selected',paint:{'line-color':'#f3f4f6','line-width':2,'line-opacity':1}},
      ],
    };
    let m:maplibregl.Map;
    try{
      m=new maplibregl.Map({container:host.current,style,center:[0,20],zoom:1.6,pitch:55,bearing:0,maxPitch:80,attributionControl:{compact:true},canvasContextAttributes:{antialias:true}});
    }catch{
      setError('WebGL could not start. Live analysis remains available through coordinate search.');return;
    }
    map.current=m;
    m.addControl(new maplibregl.NavigationControl({visualizePitch:true}),'bottom-right');
    m.addControl(new maplibregl.ScaleControl({unit:'metric'}),'bottom-left');
    const notifyCenter=()=>{const c=m.getCenter();callbacks.current.onCenterChange?.(c.lat,longitude(c.lng));};
    m.on('load',()=>{setReady(true);setError('');updateLayers();notifyCenter();if(current.current.scan)fitCurrent();});
    m.on('moveend',notifyCenter);
    m.on('click',event=>{
      if(!Number.isFinite(event.lngLat.lat)||!Number.isFinite(event.lngLat.lng))return;
      const hits=m.getLayer('hexagons')?m.queryRenderedFeatures(event.point,{layers:['hexagons']}):[];
      const id=hits[0]?.properties?.h3_index;
      if(typeof id==='string'){callbacks.current.onSelect(id);return;}
      scanAt(event.lngLat.lat,event.lngLat.lng);
    });
    m.on('mousemove',event=>{
      const hits=m.getLayer('hexagons')?m.queryRenderedFeatures(event.point,{layers:['hexagons']}):[];
      m.getCanvas().style.cursor=hits.length?'pointer':'';
    });
    m.on('error',event=>{
      const message=event.error?.message??'Map source error';
      if(/worker/i.test(message))setError('The map worker could not load. Reload this page; live API diagnostics remain available.');
      else if(/fetch|network|tile|http/i.test(message))setError('Satellite or terrain tiles failed to load. Check connectivity; available analytical layers remain visible.');
    });
    const observer=new ResizeObserver(()=>m.resize());observer.observe(host.current);
    handle.current={
      flyTo:cell=>{m.flyTo({center:[cell.longitude,cell.latitude],zoom:15.2,pitch:55,duration:1300,essential:false});},
      scanCenter:()=>{const center=m.getCenter();scanAt(center.lat,center.lng);},
      fitScan:fitCurrent,
    };
    return()=>{observer.disconnect();reticle.current?.remove();reticle.current=null;m.remove();map.current=null;handle.current=null;};
    // Map is constructed once. Current callbacks/data are held in refs so a
    // completed scan or horizon switch cannot destroy the live world canvas.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  },[]);
  useEffect(()=>{updateLayers();},[scan,selected,runout,layer]);
  useEffect(()=>{
    if(!scan||!map.current||!ready||previousScan.current===scan.scan_id)return;
    previousScan.current=scan.scan_id;fitCurrent();
  },[scan?.scan_id,ready]);
  const legend=LAYER_META[layer];
  return <section className="map-world" style={{position:'fixed',inset:0,zIndex:0}} aria-label="Live global satellite and 3D terrain map">
    <div ref={host} className="map-canvas" style={{position:'absolute',inset:0}}/>
    <div className="world-map-tools"><button onClick={()=>map.current?.easeTo({pitch:55,bearing:0})} aria-label="Reset map orientation" title="Reset map orientation"><Compass size={17}/></button>{scan&&<button onClick={fitCurrent} aria-label="Fit scan to map" title="Fit scan to map"><Maximize size={17}/></button>}</div>
    {error&&<div className="map-notice" role="status"><AlertTriangle size={14}/><span>{error}</span><button onClick={()=>setError('')} aria-label="Dismiss map notice">×</button></div>}
    {scan&&<div className="world-map-legend"><span>{legend.label}</span><div style={{height:4,background:legend.gradient}}/><div className="legend-extents"><span>{legend.left}</span><span>{legend.right}</span></div><small>Gray: data unavailable</small></div>}
  </section>;
}
