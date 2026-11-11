"""Intersect OSM segments/closed footprints with H3, retaining feature counts once per cell."""
import math
import h3


def _inside(p, ring):
    x,y=p; result=False
    for a,b in zip(ring,ring[1:]+ring[:1]):
        if (a[1]>y)!=(b[1]>y) and x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:
            result=not result
    return result


def _cross(a,b,c,d):
    def orient(p,q,r):return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    if max(a[0],b[0])+1e-12<min(c[0],d[0]) or max(c[0],d[0])+1e-12<min(a[0],b[0]):return False
    if max(a[1],b[1])+1e-12<min(c[1],d[1]) or max(c[1],d[1])+1e-12<min(a[1],b[1]):return False
    return orient(a,b,c)*orient(a,b,d)<=1e-20 and orient(c,d,a)*orient(c,d,b)<=1e-20


def intersected_cells(element,resolution=9):
    def stitch(parts):
        parts=[list(p) for p in parts if p];rings=[]
        while parts:
            chain=parts.pop()
            while chain[0]!=chain[-1]:
                match=None
                for i,p in enumerate(parts):
                    if chain[-1]==p[0]:match=(i,p[1:]);break
                    if chain[-1]==p[-1]:match=(i,list(reversed(p))[1:]);break
                if match is None:break
                i,tail=match;chain.extend(tail);parts.pop(i)
            rings.append(chain)
        return rings
    geometries=[]
    if element.get('geometry'):geometries.append(element['geometry'])
    geometries.extend(stitch(m['geometry'] for m in element.get('members',[]) if m.get('geometry') and m.get('role')!='inner'))
    holes=stitch(m['geometry'] for m in element.get('members',[]) if m.get('geometry') and m.get('role')=='inner')
    if not geometries and element.get('lat') is not None:
        return {h3.latlng_to_cell(element['lat'],element['lon'],resolution)}
    result=set()
    for geometry in geometries:
        points=[(float(p['lon']),float(p['lat'])) for p in geometry]
        if len(points)<2:continue
        lon0=points[0][0]
        unwrap=lambda x:lon0+(x-lon0+180)%360-180
        points=[(unwrap(x),y) for x,y in points]
        candidates=set()
        for a,b in zip(points,points[1:]):
            distance=h3.great_circle_distance((a[1],a[0]),(b[1],b[0]),unit='m')
            for i in range(max(1,math.ceil(distance/80))+1):
                t=i/max(1,math.ceil(distance/80));x=a[0]+t*(b[0]-a[0]);y=a[1]+t*(b[1]-a[1])
                candidates.update(h3.grid_disk(h3.latlng_to_cell(y,(x+180)%360-180,resolution),1))
        polygon=points[0]==points[-1] and len(points)>3
        if polygon:
            candidates.update(h3.polygon_to_cells(h3.LatLngPoly([(y,(x+180)%360-180) for x,y in points[:-1]]),resolution))
        for cell in candidates:
            ring=[(unwrap(lon),lat) for lat,lon in h3.cell_to_boundary(cell)]
            interior_hole=False
            for hole in holes:
                h=[(unwrap(float(p['lon'])),float(p['lat'])) for p in hole]
                if len(h)>3 and h[0]==h[-1] and all(_inside(p,h) for p in ring) and not any(
                    _cross(a,b,c,d) for a,b in zip(h,h[1:]) for c,d in zip(ring,ring[1:]+ring[:1])):
                    interior_hole=True;break
            if interior_hole:continue
            if any(_inside(p,ring) for p in points) or (polygon and any(_inside(p,points) for p in ring)) or any(
                _cross(a,b,c,d) for a,b in zip(points,points[1:]) for c,d in zip(ring,ring[1:]+ring[:1])):
                result.add(cell)
    return result
