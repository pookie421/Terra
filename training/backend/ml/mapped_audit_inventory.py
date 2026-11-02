"""Additional cells strictly inside archived NASA polygons; never buffer report points."""
import json
from collections import Counter

def mapped_candidates(catalog_path):
    from backend.ml.build_real_labeled_dataset import EventCandidate, parse_date, _mapped_polygon_h3_cells
    if not catalog_path.exists():
        return [], {'status':'unavailable', 'reason':'No archived mapped source catalog'}
    catalog=json.loads(catalog_path.read_text())
    result=[];seen=set();counts=Counter();skips=Counter()
    for layer,entry in catalog.items():
        for feature in entry['response'].get('features',[]):
            p=feature['properties']; trigger=str(p.get('landslide_trigger','')).lower()
            if trigger=='earthquake': regime='coseismic'
            elif trigger in ('snowfall_snowmelt','freeze_thaw'):regime='cryosphere'
            elif p.get('landslide_setting')=='burned_area':regime='post_wildfire'
            elif trigger=='construction':regime='anthropogenic_cut_mining'
            elif trigger in ('rain','downpour','continuous_rain','monsoon','tropical_cyclone'):regime='monsoon_rain'
            else:skips['unclassified_trigger']+=1;continue
            day=parse_date(str(p.get('event_date')))
            citation=p.get('source_link') or p.get('citation')
            if not day or not citation or citation=='<Null>':skips['missing_date_or_citation']+=1;continue
            if p.get('landslide_category')=='snow_avalanche':skips['snow_avalanche']+=1;continue
            geometry=feature['geometry'];cells=_mapped_polygon_h3_cells(geometry)
            if not cells:skips['no_h3_center_inside_polygon']+=1;continue
            # One source inventory/day is one group, including all its footprints.
            group='mapped_inventory:'+str(p.get('country_name'))+':'+day
            source='nasa_mapped:'+layer+':'+str(p['objectid'])
            for cell,lat,lon in cells:
                key=(group,cell)
                if key in seen or counts[regime]>=80:continue
                seen.add(key);counts[regime]+=1
                result.append(EventCandidate(event_id=source+':'+cell,name=p.get('event_title') or source,
                    event_date=day,latitude=lat,longitude=lon,regime=regime,
                    source_name='NASA COOLR archived mapped polygon',source_url=entry['source_url'].split('?')[0]+'?f=geojson&objectIds='+str(p['objectid']),
                    event_record_type='audit_mapped_polygon_cell',label=1,label_source='H3 center inside sourced failure footprint',
                    label_quality='mapped_failure_footprint_not_verified_initiation_crown',citation=citation,
                    failure_geometry=geometry,h3_cell_id=cell,source_event_id=source,event_group_id=group,
                    mapping_method=p.get('method') or 'COOLR report polygon'))
    return result, {'status':'archived_source_geometry','added_positive_cells_by_regime':dict(counts),
        'excluded_source_records':dict(skips),'positive_balance_achieved':len(counts)==5 and max(counts.values())/min(counts.values())<=2,
        'limitation':'Rare regimes lack enough dated cited mapped polygons in the queried source catalog. Point reports are not expanded into independent failures; cells share inventory groups. Footprint labels can include runout, not only initiation.'}
