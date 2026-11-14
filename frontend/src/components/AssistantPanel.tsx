import {useEffect,useRef,useState,type RefObject} from 'react';
import {AlertCircle,ArrowDown,ArrowUpRight,Check,Circle,LoaderCircle,Send,Square} from 'lucide-react';
import type {Action,ActionOutcome,BoundClaim,Cell,NasaProduct,PredictionModel,Scan,ThoughtStep} from '../types';
import {fmt} from '../types';
import {modelLabel} from '../modelPresentation';
import {assistantSuggestions,assistantTasks,inlineParts,proseBlocks,taskResultSummary,type TaskStatus} from '../assistantPresentation';

export interface AssistantMessage {
  id?:string;role:'assistant'|'user';text:string;guard?:string;claims?:BoundClaim[];
  scanId?:string|null;state?:string|null;cell?:string;model?:PredictionModel;nasaProduct?:NasaProduct;
  assistantProvider?:string;assistantModel?:string;error?:string;steps?:ThoughtStep[];
  plannedActions?:Action[];outcomes?:ActionOutcome[];executing?:boolean;activeTask?:number;
  executionError?:string;cancelled?:boolean;
}
export interface AssistantPanelProps {
  scan:Scan|null;selectedCell?:Cell;predictionModel:PredictionModel;nasaProduct:NasaProduct;
  messages:AssistantMessage[];draft:string;busy:boolean;workspaceBusy?:boolean;
  connectionLabel:string;connected?:boolean;hasKey:boolean;
  onDraftChange:(text:string)=>void;onSend:()=>void;onStop:()=>void;onOpenOptions:()=>void;
  inputRef?:RefObject<HTMLTextAreaElement>;endRef?:RefObject<HTMLDivElement>;
}

const statusLabels:Record<TaskStatus,string>={planned:'Planned',running:'In progress',completed:'Completed',failed:'Failed',cancelled:'Not completed'};
function InlineText({text}:{text:string}){
  return <>{inlineParts(text).map((part,index)=>{
    if(part.kind==='strong')return <strong key={index}>{part.text}</strong>;
    if(part.kind==='emphasis')return <em key={index}>{part.text}</em>;
    if(part.kind==='code')return <code key={index}>{part.text}</code>;
    if(part.kind==='link')return <a key={index} href={part.href} target="_blank" rel="noopener noreferrer">{part.text}</a>;
    return part.text;
  })}</>;
}
function AssistantProse({text}:{text:string}){
  return <div className="assistant-prose">{proseBlocks(text).map((block,index)=>{
    if(block.kind==='heading')return <h3 key={index}><InlineText text={block.text}/></h3>;
    if(block.kind==='code')return <pre key={index}><code>{block.text}</code></pre>;
    if(block.kind==='list'){
      const items=block.items.map((item,itemIndex)=><li key={itemIndex} value={block.ordered?block.numbers?.[itemIndex]:undefined}><InlineText text={item}/></li>);
      return block.ordered?<ol key={index}>{items}</ol>:<ul key={index}>{items}</ul>;
    }
    if(block.kind==='table')return <div className="assistant-table-wrap" key={index}><table><thead><tr>{block.headers.map((header,column)=><th key={column} scope="col"><InlineText text={header}/></th>)}</tr></thead><tbody>{block.rows.map((row,rowIndex)=><tr key={rowIndex}>{row.map((value,column)=><td key={column}><InlineText text={value}/></td>)}</tr>)}</tbody></table></div>;
    return <p key={index}><InlineText text={block.text}/></p>;
  })}</div>;
}
function StatusIcon({status}:{status:TaskStatus}){
  if(status==='completed')return <Check size={14}/>;
  if(status==='running')return <LoaderCircle size={14} className="spin"/>;
  if(status==='failed')return <AlertCircle size={14}/>;
  if(status==='cancelled')return <Square size={11}/>;
  return <Circle size={12}/>;
}
function Tasks({message}:{message:AssistantMessage}){
  const tasks=assistantTasks(message),summary=taskResultSummary(message);
  if(!tasks.length)return null;
  return <section className="assistant-task-card" aria-label="Workspace tasks">
    <p className="assistant-result-summary" role={message.executing?'status':undefined}>{summary}</p>
    <ol className="assistant-task-list">{tasks.map(task=><li key={task.step} className={`task-${task.status}`}>
      <span className="task-status-icon" aria-hidden="true"><StatusIcon status={task.status}/></span>
      <div className="task-copy"><strong>{task.label}</strong><small>{statusLabels[task.status]}</small>{task.status==='failed'&&task.detail&&<p>{task.detail}</p>}</div>
    </li>)}</ol>
    {message.cancelled&&<p className="support-note">Remaining actions were cancelled. A scan already sent to the server may still finish.</p>}
    {message.executionError&&!message.cancelled&&<p className="execution-error" role="status">{message.executionError}</p>}
  </section>;
}
function verification(message:AssistantMessage):string {
  if(message.guard==='VERIFIED_LLM_PROSE')return 'Verified AI response';
  if(message.guard==='VERIFIED_TEMPLATE')return 'Verified map command';
  if(message.guard)return 'Response could not be verified';
  return 'Assistant';
}
function earlierContext(message:AssistantMessage,props:AssistantPanelProps):boolean {
  if(!props.scan)return !!message.scanId;
  return !!((message.scanId&&message.scanId!==props.scan.scan_id)||
    (message.state&&message.state!==props.scan.scenario_id)||
    (message.model&&message.model!==props.predictionModel)||
    (message.cell&&props.selectedCell&&message.cell!==props.selectedCell.h3_index)||
    (message.model==='nasa_lhasa_2'&&message.nasaProduct&&message.nasaProduct!==props.nasaProduct));
}
function MessageDetails({message,previousContext}:{message:AssistantMessage;previousContext:boolean}){
  const details=message.scanId||message.state||message.error||message.claims?.length||message.outcomes?.length||message.assistantProvider;
  return <>
    {!!message.steps?.length&&<details className="assistant-step-details"><summary>Evidence &amp; action notes</summary><p className="support-note">Public evidence and action summaries.</p><ol>{message.steps.map((step,index)=><li key={`${step.step}-${index}`} value={step.step}><strong>{step.phase}</strong><AssistantProse text={step.reasoning}/></li>)}</ol></details>}
    {details&&<details className="assistant-response-details"><summary>{previousContext?'Response details · earlier context':'Response details & numeric citations'}</summary>
      {previousContext&&<p className="support-note">This response describes the scan, model and selection recorded below.</p>}
      {message.guard&&<p><code>{message.guard}</code></p>}
      {message.assistantProvider&&<p>AI provider: {message.assistantProvider}{message.assistantModel?' · '+message.assistantModel:''}</p>}
      {message.error&&<p className="execution-error">Assistant service: {message.error}</p>}
      {(message.scanId||message.state||message.cell||message.model)&&<dl className="assistant-evidence-context">
        {message.scanId&&<><dt>Scan</dt><dd><code>{message.scanId}</code></dd></>}
        {message.state&&<><dt>Evidence state</dt><dd><code>{message.state}</code></dd></>}
        {message.cell&&<><dt>Cell</dt><dd><code>{message.cell}</code></dd></>}
        {message.model&&<><dt>Model</dt><dd>{modelLabel(null,message.model)}{message.model==='nasa_lhasa_2'&&message.nasaProduct?' · NASA '+(message.nasaProduct==='today'?'Today':'Tomorrow'):''}</dd></>}
      </dl>}
      {!!message.claims?.length&&<div className="assistant-claims" aria-label="Verified numeric bindings">{message.claims.map((claim,index)=><div className="assistant-claim" key={`${claim.path}-${index}`}><span>{claim.name.replaceAll('_',' ')}</span><b title={String(claim.value)}>{fmt(claim.value,5)}</b><code>{claim.path}</code></div>)}</div>}
      {message.outcomes?.map((outcome,index)=><div className="assistant-outcome-detail" key={`${outcome.step}-${index}`}><p>Task {outcome.step}: {outcome.summary}</p>{(outcome.scan_id||outcome.scenario_id)&&<p className="support-note">Scan: <code>{outcome.scan_id??'Not reported'}</code><br/>State: <code>{outcome.scenario_id??'Not reported'}</code></p>}</div>)}
    </details>}
  </>;
}

export default function AssistantPanel(props:AssistantPanelProps){
  const {scan,predictionModel,nasaProduct,messages,draft,busy,workspaceBusy,connectionLabel,connected,hasKey,onDraftChange,onSend,onStop,onOpenOptions}=props;
  const transcript=useRef<HTMLDivElement>(null),ownInput=useRef<HTMLTextAreaElement>(null),followLatest=useRef(true),previousCount=useRef(0);
  const input=props.inputRef??ownInput;
  const [showLatest,setShowLatest]=useState(false);
  const suggestions=assistantSuggestions(scan,predictionModel,nasaProduct);
  const blocked=busy||workspaceBusy||!draft.trim();
  useEffect(()=>{
    const appended=messages.length!==previousCount.current;previousCount.current=messages.length;
    const log=transcript.current;if(!followLatest.current||!log)return;
    if(!messages.length){log.scrollTop=0;return;}
    if(appended&&messages.at(-1)?.role==='assistant'){
      const reply=log.querySelector<HTMLElement>('article.assistant-message:last-of-type');
      if(reply)log.scrollTop+=reply.getBoundingClientRect().top-log.getBoundingClientRect().top;
    }else if(messages.at(-1)?.role!=='assistant')log.scrollTop=log.scrollHeight;
  },[messages,busy]);
  useEffect(()=>{
    const field=input.current;
    if(field){field.style.height='auto';field.style.height=Math.min(field.scrollHeight,144)+'px';}
  },[draft,input]);
  function chooseSuggestion(prompt:string){onDraftChange(prompt);input.current?.focus();}
  function sendMessage(){if(blocked)return;followLatest.current=true;setShowLatest(false);onSend();}
  function latest(){followLatest.current=true;setShowLatest(false);if(transcript.current)transcript.current.scrollTop=transcript.current.scrollHeight;}
  const pending=busy&&!messages.some(message=>message.executing);
  return <div className="assistant-panel">
    <div className="assistant-context-line">{scan?<><span>{modelLabel(scan,predictionModel)}</span>{predictionModel==='nasa_lhasa_2'&&<span>NASA {nasaProduct==='today'?'Today':'Tomorrow'}</span>}</>:<span>Explore a location or navigate the map</span>}</div>
    <div className="assistant-connection-row"><span className={`assistant-mode-tag ${hasKey?'configured':''}`} title={hasKey?connected?'This configuration returned a verified AI response.':'A key is configured; it has not yet returned a verified response.':'Map commands work without an AI key.'}>{hasKey?(connected?'Connected · ':'Key configured · ')+connectionLabel:'Map commands · no AI key'}</span><button className="assistant-options-link" onClick={onOpenOptions}>{hasKey?'AI settings in Options':'Set up AI in Options'}</button></div>
    <div className="assistant-transcript" ref={transcript} role="log" aria-label="Assistant conversation" aria-live="polite" aria-relevant="additions text" onScroll={()=>{const element=transcript.current;if(element){followLatest.current=element.scrollHeight-element.scrollTop-element.clientHeight<80;setShowLatest(!followLatest.current);}}}>
      {!messages.length&&<div className="assistant-welcome"><h3>Work with the map</h3><p>Ask about the evidence or describe a sequence of map actions. Suggestions prepare a message for you to review.</p><div className="assistant-suggestions">{suggestions.map(suggestion=><button key={suggestion.id} className="assistant-suggestion" onClick={()=>chooseSuggestion(suggestion.prompt)}><strong>{suggestion.label}<ArrowUpRight size={13}/></strong><span>{suggestion.description}</span></button>)}</div></div>}
      {messages.map((message,index)=><article className={`assistant-message ${message.role}`} key={message.id??index}>
        <div className="assistant-message-meta">{message.role==='user'?<span>You</span>:<span className={`assistant-verification ${message.guard==='VERIFIED_LLM_PROSE'?'verified':message.guard==='VERIFIED_TEMPLATE'?'template':message.guard?'failed':''}`} title={message.guard}>{message.guard==='VERIFIED_LLM_PROSE'||message.guard==='VERIFIED_TEMPLATE'?<Check size={12}/>:message.guard?<AlertCircle size={12}/>:null}{verification(message)}</span>}</div>
        {message.role==='assistant'&&<Tasks message={message}/>}
        <AssistantProse text={message.text}/>
        {message.role==='assistant'&&<MessageDetails message={message} previousContext={earlierContext(message,props)}/>}
      </article>)}
      {pending&&<div className="assistant-pending" role="status"><LoaderCircle size={14} className="spin"/><span>Waiting for the assistant response…</span></div>}
      <div ref={props.endRef}/>
    </div>
    {showLatest&&<button className="assistant-latest" onClick={latest}>Jump to latest<ArrowDown size={13}/></button>}
    {!!messages.length&&!busy&&<details className="assistant-followups"><summary>Suggested tasks</summary><div className="assistant-suggestions">{suggestions.map(suggestion=><button key={suggestion.id} className="assistant-suggestion" onClick={()=>chooseSuggestion(suggestion.prompt)}><strong>{suggestion.label}<ArrowUpRight size={13}/></strong><span>{suggestion.description}</span></button>)}</div></details>}
    <form className="assistant-composer" onSubmit={event=>{event.preventDefault();sendMessage();}}><textarea ref={input} rows={2} aria-label="Ask about evidence or describe map actions" value={draft} onChange={event=>onDraftChange(event.target.value)} onKeyDown={event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.nativeEvent.isComposing){event.preventDefault();sendMessage();}}} placeholder="Ask about this area, or describe the next steps…"/>
      <div className="assistant-composer-actions"><span className="assistant-composer-hint">{workspaceBusy&&!busy?'A scan is running. You can prepare your next message.':'Enter to send · Shift + Enter for a new line'}</span>{busy&&<button type="button" onClick={onStop} title="Cancel the reply and remaining actions. A scan already sent to the server may still finish."><Square size={12}/>Stop</button>}<button className="assistant-send" type="submit" disabled={!!blocked} aria-label="Send message"><Send size={15}/></button></div>
    </form>
  </div>;
}
