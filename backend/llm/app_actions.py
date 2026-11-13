"""Validated, state-bound application commands and public execution summaries.

This module is independent of the language model. Plans describe requested UI
work; they never assert that a scan, forecast or camera action already ran.
"""
import hashlib
import json
import math
import re

MODELS = ('terra_physics_ml', 'nasa_lhasa_2', 'physics_only', 'weather_terrain_ml', 'rainfall_only')
HORIZONS = ('now', 't_24h', 't_48h', 't_72h', 'peak_72h')
LAYERS = ('probability', 'fs_mid', 'lhasa_2', 'nisar_radar', 'slope', 'pore_pressure')
TARGETS = ('steepest', 'riskiest', 'selected', 'lowest_fs', 'flattest')
ACTION_TYPES = ('scan_coordinates', 'search_location', 'select_prediction_model', 'set_time_horizon',
                'set_map_layer', 'fly_to_cell', 'inspect_cell', 'compare_cells')
SCHEMA_VERSION = 'agentic_app_v2'
H3_TOKEN = re.compile(r'(?<![\w.])8[0-9a-f]{14}(?![\w.])',re.I)


def _number(value, low, high, name):
    try:finite=isinstance(value,(int,float)) and math.isfinite(value)
    except (ValueError,OverflowError):finite=False
    if isinstance(value, bool) or not finite or not low <= value <= high:
        raise ValueError('Invalid ' + name)
    return float(value)


def cell_registry(scan):
    if not scan:
        return {}
    cells = {str(k): dict(v) for k, v in scan.get('cell_registry', {}).items()}
    for feature in scan.get('h3_geojson', {}).get('features', []):
        p = feature.get('properties') or {}
        cid = p.get('h3_index') or p.get('cell_id')
        if cid:
            cells[cid] = {**cells.get(cid, {}), **p}
    for section in ('physical_assessment', 'physics_informed_assessment'):
        for p in scan.get(section, {}).get('cells', []):
            cid = p.get('h3_index')
            if cid:
                cells[cid] = {**cells.get(cid, {}), **p}
    selected = scan.get('selected_cell') or {}
    if selected.get('h3_index'):
        cid = selected['h3_index']; cells[cid] = {**cells.get(cid, {}), **selected}
    for record in (scan.get('nasa_lhasa') or {}).get('cells',[]):
        cid=record.get('h3_index')
        if cid in cells:cells[cid]['lhasa_probability']=record.get('official_score')
    return cells


def resolve_target(target, scan, selected=None, active_prediction_model='terra_physics_ml'):
    if target not in TARGETS:
        raise ValueError('Unknown cell selector')
    cells = cell_registry(scan)
    if not cells:
        raise ValueError('No cells in this state')
    if target == 'selected':
        chosen = selected or (scan.get('selected_cell') or {}).get('h3_index') or scan.get('summary', {}).get('hotspot_h3_index')
        if chosen in cells:
            return chosen
        raise ValueError('Selected cell is not in this state')
    if target=='riskiest' and active_prediction_model=='nasa_lhasa_2':
        official=[]
        for cid,record in cells.items():
            value=record.get('lhasa_probability',record.get('official_score'))
            if isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value) and 0<=value<=1:
                official.append((cid,value))
        if official:
            # Official probabilities and unvalidated proxy indices are separate
            # cohorts. A high fallback index must not outrank a NASA probability.
            return sorted(official,key=lambda item:(-item[1],item[0]))[0][0]
    field = 'fs_mid' if target == 'lowest_fs' else 'score' if target == 'riskiest' else 'slope_deg'
    def value(c):
        if field=='score' and active_prediction_model!='terra_physics_ml':
            return (c.get('prediction_scores') or {}).get(active_prediction_model)
        v = c.get(field)
        if v is None and field == 'score':
            v = c.get('physics_informed_score', c.get('p_failure'))
        return v
    eligible = [(cid, value(c)) for cid, c in cells.items() if isinstance(value(c), (float, int)) and not isinstance(value(c), bool) and math.isfinite(value(c))]
    if not eligible:
        raise ValueError('Cell selector has no finite evidence: ' + target)
    return sorted(eligible, key=lambda item: ((-item[1] if target in ('steepest', 'riskiest') else item[1]), item[0]))[0][0]


def validate_actions(actions, scan=None, scan_id=None, scenario_id=None, selected=None):
    """Validate an entire chain before executing any part of it.

    A new scan changes the cell universe. Following selectors are deliberately
    left symbolic and resolved by the executor against the newly returned grid.
    """
    if not isinstance(actions, list) or len(actions) > 20:
        raise ValueError('An action chain must contain at most twenty commands')
    if scan and scan_id is None and scenario_id is None:
        scan_id=scan.get('scan_id');scenario_id=scan.get('scenario_id')
    if (scan_id is None)!=(scenario_id is None):raise ValueError('Both originating state identifiers are required')
    if scan and (scan.get('scan_id'),scan.get('scenario_id'))!=(scan_id,scenario_id):
        raise ValueError('Action evidence belongs to another scan or state')
    result = []; chain = False; new_location = False; cells = cell_registry(scan)
    active_model=(scan or {}).get('active_prediction_model','terra_physics_ml')
    common = {'type', 'scan_id', 'scenario_id', 'state_scope'}
    for raw in actions:
        if not isinstance(raw, dict) or raw.get('type') not in ACTION_TYPES:
            raise ValueError('Unknown app action')
        kind = raw['type']
        if raw.get('scan_id', scan_id) != scan_id or raw.get('scenario_id', scenario_id) != scenario_id:
            raise ValueError('Action belongs to another scan or state')
        expected_scope = 'chain' if chain else 'bound'
        if raw.get('state_scope', expected_scope) != expected_scope:
            raise ValueError('Action has an invalid state scope')
        out = {'type': kind, 'scan_id': scan_id, 'scenario_id': scenario_id, 'state_scope': expected_scope}
        allowed = set()
        if kind == 'scan_coordinates':
            allowed = {'latitude', 'longitude', 'label'}
            out.update(latitude=_number(raw.get('latitude'), -90, 90, 'latitude'),
                       longitude=_number(raw.get('longitude'), -180, 180, 'longitude'))
            label = raw.get('label', '')
            if not isinstance(label, str) or len(label) > 240:
                raise ValueError('Invalid coordinate label')
            out['label'] = label.strip(); chain = True; new_location = True
        elif kind == 'search_location':
            allowed = {'query'}; query = raw.get('query')
            if not isinstance(query, str) or not query.strip() or len(query) > 240:
                raise ValueError('Invalid location query')
            out['query'] = query.strip(); chain = True; new_location = True
        elif kind in ('select_prediction_model', 'set_time_horizon', 'set_map_layer'):
            field, enum = {'select_prediction_model': ('model', MODELS), 'set_time_horizon': ('horizon', HORIZONS), 'set_map_layer': ('layer', LAYERS)}[kind]
            allowed = {field}
            if raw.get(field) not in enum:
                raise ValueError('Invalid ' + field)
            out[field] = raw[field]
            if kind == 'select_prediction_model':active_model=raw[field]
            if kind == 'set_time_horizon':
                chain = True
        else:
            allowed = {'h3_index', 'target'} if kind != 'compare_cells' else {'primary_h3', 'secondary_h3', 'primary_target', 'secondary_target'}
            pairs = (('h3_index', 'target'),) if kind != 'compare_cells' else (('primary_h3', 'primary_target'), ('secondary_h3', 'secondary_target'))
            for fixed, symbolic in pairs:
                if (fixed in raw) == (symbolic in raw):
                    raise ValueError('A cell action needs exactly one cell ID or selector')
                if symbolic in raw:
                    if raw[symbolic] not in TARGETS:
                        raise ValueError('Invalid cell selector')
                    if chain:
                        out[symbolic] = raw[symbolic]
                    else:
                        out[fixed] = resolve_target(raw[symbolic], scan, selected,active_model)
                else:
                    cid = raw[fixed]
                    if not isinstance(cid, str) or not re.fullmatch(r'[0-9a-f]{15}', cid) or cid not in cells:
                        raise ValueError('Cell is not in the bound scan state')
                    if new_location:
                        raise ValueError('Use a selector after an action that changes the scan or horizon')
                    out[fixed] = cid
            if kind == 'compare_cells' and out.get('primary_h3') is not None and out.get('primary_h3') == out.get('secondary_h3'):
                raise ValueError('Cell comparison requires two distinct cells')
            if kind == 'fly_to_cell':
                allowed |= {'pitch', 'bearing', 'zoom'}
                out.update(pitch=_number(raw.get('pitch', 55.), 0, 80, 'pitch'),
                           bearing=_number(raw.get('bearing', 0.), -360, 360, 'bearing'),
                           zoom=_number(raw.get('zoom', 13.), 0, 22, 'zoom'))
        if set(raw) - common - allowed:
            raise ValueError('Unsupported action fields')
        result.append(out)
    return result


def plan_actions(message, scan=None, selected=None):
    """Extract explicit app intent in textual order without inventing forcing."""
    if not isinstance(message, str) or not message.strip() or len(message) > 6000:
        raise ValueError('Message must contain at most six thousand characters')
    events = []; text = message.lower(); covered = []
    coordinate = re.compile(r'(?<![\w.])(?P<lat>[+-]?\d+(?:\.\d+)?)\s*,\s*(?P<lon>[+-]?\d+(?:\.\d+)?)(?![\d.])')
    for match in coordinate.finditer(message):
        prefix = text[max(0, match.start()-65):match.start()]
        if not re.search(r'(?:scan|search(?: for)?|take me to|go to|fly to|look at|analy[sz]e|coordinates?\s*:?|at)\s*$', prefix) and message.strip() != match.group():
            continue
        action={'type':'search_location','query':match.group()} if re.search(r'search(?: for)?\s*$',prefix) else {'type': 'scan_coordinates', 'latitude': float(match['lat']), 'longitude': float(match['lon']), 'label': match.group()}
        events.append((match.start(),action)); covered.append(match.span())
    named = re.compile(r'\b(?:search(?: for)?|scan|take me to|go to|fly to|analy[sz]e)\s+(.+?)(?=,\s*(?:then|switch|compare|inspect|show)|\s+(?:and then|then|and switch|and inspect)|[;.!]|$)', re.I)
    for match in named.finditer(message):
        if any(a <= match.end() and b >= match.start() for a,b in covered):
            continue
        query = match.group(1).strip(' ,')
        if re.search(r'^(?:the )?(?:selected|current|active|riskiest|steepest|critical|this|that|cell|area|hotspot)\b', query, re.I):
            continue
        if query:
            events.append((match.start(), {'type': 'search_location', 'query': query}))
    patterns = (
        (r'\b(?:nasa(?:\s+lhasa)?(?:\s*2(?:\.0)?)?|lhasa(?:\s*2(?:\.0)?)?)\b', 'nasa_lhasa_2'),
        (r'\b(?:terra(?:[- ]shield)?|in[- ]house(?:\s+physics\s*\+\s*ml)?)\b', 'terra_physics_ml'),
        (r'\bphysics[- ]only\b', 'physics_only'),
        (r'\bweather\s*(?:\+|and)\s*terrain(?:\s+ml)?\b', 'weather_terrain_ml'),
        (r'\brainfall[- ]only\b', 'rainfall_only'))
    for pattern, model in patterns:
        for match in re.finditer(pattern, message, re.I):
            prefix = text[max(0, match.start()-70):match.start()]
            if re.search(r'\b(?:switch|select|use|back to|change to|compare)\b', prefix) or 'compare' in text:
                events.append((match.start(), {'type': 'select_prediction_model', 'model': model}))
    for match in re.finditer(r'\bpeak(?:[ -]*(?:over|of|next))?(?:\s*72\s*(?:h|hours?))?\b|(?:\bt\s*\+?\s*|\+)?\b(24|48|72)\s*(?:h(?:ours?)?|hours?)\b|\b(?:now|tomorrow)\b', message, re.I):
        value = match.group().lower()
        if value == 'now' and not re.search(r'\b(?:show|switch|back to|return to|set|view)\b', text[max(0, match.start()-35):match.start()]):
            continue
        horizon = 'peak_72h' if value.startswith('peak') else 't_24h' if value == 'tomorrow' else 'now' if value == 'now' else 't_'+match.group(1)+'h'
        events.append((match.start(), {'type': 'set_time_horizon', 'horizon': horizon}))
    layer_patterns = ((r'\b(?:nisar(?:\s+(?:l-sar|radar))?|radar)(?:\s+layer)?\b','nisar_radar'), (r'\b(?:pore[- ]pressure|pressure head)(?:\s+layer)?\b','pore_pressure'),
                      (r'\b(?:factor[- ]of[- ]safety|fs_mid|fs)(?:\s+layer)?\b','fs_mid'), (r'\bslope(?:\s+layer)?\b','slope'),
                      (r'\b(?:probability|initiation score)(?:\s+layer)?\b','probability'))
    for pattern, layer in layer_patterns:
        for match in re.finditer(pattern, message, re.I):
            prefix = text[max(0, match.start()-30):match.start()]
            if re.search(r'\b(?:layer|colou?r|show|map)\b', prefix) or 'layer' in match.group().lower():
                events.append((match.start(), {'type':'set_map_layer','layer':layer}))
    selectors = []
    for pattern, target in ((r'\b(?:steepest(?:\s+cliff)?|cliff)\b','steepest'),(r'\b(?:riskiest|highest risk|critical cell|hotspot)\b','riskiest'),
                            (r'\b(?:lowest fs|lowest factor of safety)\b','lowest_fs'),(r'\b(?:valley floor|flattest|flat cell|lowest slope)\b','flattest')):
        selectors += [(m.start(),target) for m in re.finditer(pattern,message,re.I)]
    ids = [(m.start(), m.group().lower()) for m in H3_TOKEN.finditer(message)]
    selectors.sort(); ids.sort()
    if 'compare' in text and len({v for _,v in selectors}) >= 2:
        unique = list(dict.fromkeys(v for _,v in selectors))
        events.append((max(p for p,_ in selectors), {'type':'compare_cells','primary_target':unique[0],'secondary_target':unique[1]}))
    elif 'compare' in text and len(ids) >= 2:
        events.append((ids[-1][0], {'type':'compare_cells','primary_h3':ids[0][1],'secondary_h3':ids[1][1]}))
    elif scan or any(a['type'] in ('scan_coordinates','search_location') for _,a in events):
        target = {'h3_index':ids[-1][1]} if ids else {'target':selectors[-1][1] if selectors else 'selected'}
        point = max((p for p,_ in selectors+ids), default=len(message))
        if re.search(r'\b(?:fly|zoom|locate)\b|\bshow\s+(?:me\s+)?(?:the\s+)?(?:selected\s+)?(?:cell|cliff|hotspot|steepest)\b',text):
            events.append((point-.1, {'type':'fly_to_cell',**target}))
        events.append((point, {'type':'inspect_cell',**target}))
    events.sort(key=lambda item:item[0])
    raw = [action for _,action in events]
    return validate_actions(raw, scan, (scan or {}).get('scan_id'), (scan or {}).get('scenario_id'), selected)


def action_envelope(actions, scan_id, scenario_id):
    identity = json.dumps([scan_id, scenario_id, actions], sort_keys=True, separators=(',',':'))
    return {'schema_version':SCHEMA_VERSION,'plan_id':hashlib.sha256(identity.encode()).hexdigest()[:24],
            'execution_status':'PLANNED','actions':actions}


def public_steps(actions, has_evidence):
    """Short public summaries, never private deliberation or measured claims."""
    return [
        {'step':1,'phase':'Intent and spatial target','reasoning':'The requested location, model, horizon and cell controls were parsed into an ordered app plan.'},
        {'step':2,'phase':'Available evidence','reasoning':('The plan is bound to the active scan and selected-cell evidence, including disclosed source availability.' if has_evidence else 'A location scan must complete before physical, NASA comparator or radar evidence can be inspected.')},
        {'step':3,'phase':'App control plan','reasoning':('The app will execute the validated commands in order and resolve cell selectors against each returned state.' if actions else 'No app change is planned; the existing analysis remains selected.')}
    ]
