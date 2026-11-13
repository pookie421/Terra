import {useEffect,useState} from 'react';
import {ExternalLink,LoaderCircle} from 'lucide-react';
import {api} from '../api';
import type {Connections} from '../types';

export default function ConnectionStatus(){
  const [connections,setConnections]=useState<Connections>(),[error,setError]=useState(''),[attempt,setAttempt]=useState(0);
  useEffect(()=>{
    const controller=new AbortController();setError('');
    api.connections(controller.signal).then(value=>{if(!controller.signal.aborted)setConnections(value);}).catch(()=>{if(!controller.signal.aborted)setError('Connection settings could not be read.');});
    return()=>controller.abort();
  },[attempt]);
  return <>
    <p className="support-note">Shared deployment users do not need personal API keys. The deployment owner configures the assistant and optional Earthdata access on the server.</p>
    {!connections&&!error&&<p className="connection-loading" role="status"><LoaderCircle size={13} className="spin"/>Reading configuration…</p>}
    {error&&<p className="support-note amber" role="status">{error} <button className="inline-button" onClick={()=>setAttempt(v=>v+1)}>Retry</button></p>}
    {connections&&<dl className="telemetry connection-settings"><div><dt>Assistant</dt><dd>{connections.assistant.configured?'Configured':'Not configured'}</dd></div><div><dt>Earthdata</dt><dd>{connections.earthdata.configured?'Configured':'Not configured'}</dd></div></dl>}
    <p className="support-note">Configuration status does not confirm service availability. Provider results and errors appear in Sources and assistant responses.</p>
    {connections?.assistant.mode==='hosted'&&<p className="support-note">The hosted assistant receives the selected scan context and your messages.</p>}
    <details className="nested-details"><summary>Owner setup and help</summary>
      {connections&&<dl className="telemetry"><div><dt>Assistant provider</dt><dd>{connections.assistant.provider==='ollama'?'Local Ollama':connections.assistant.provider==='openai_compatible'?'Compatible API':'Setup needed'}</dd></div><div><dt>Model</dt><dd>{connections.assistant.model??(connections.assistant.mode==='local'?'Selected automatically':'Not configured')}</dd></div><div><dt>Mode</dt><dd>{connections.assistant.mode.replaceAll('_',' ')}</dd></div></dl>}
      <p className="support-note">Configure connections in the server environment. Keys stay on the server and are never entered or stored in this browser.</p>
      <a className="text-link" href="https://github.com/pookie421/Terra/blob/main/docs/assistant_connections.md" target="_blank" rel="noreferrer">Deployment setup guide<ExternalLink size={13}/></a>
    </details>
  </>;
}
