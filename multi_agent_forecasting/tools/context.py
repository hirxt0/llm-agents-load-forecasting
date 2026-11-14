import json
import re
import time
import urllib.parse
import urllib.request
import zlib
from functools import lru_cache
import holidays
import numpy as np
import pandas as pd
from config import RAW, CONTEXT_RAW, CACHE
SITES = {'Bear': dict(country='US', subdiv='CA', city='Berkeley, California, USA', utc_offset=-8, climate='Mediterranean (mild, dry summers; cool, wet winters)', tz='US/Pacific'), 'Fox': dict(country='US', subdiv='AZ', city='Tempe, Arizona, USA', utc_offset=-7, climate='hot desert (very hot summers, mild winters)', tz='US/Mountain'), 'Panther': dict(country='US', subdiv='FL', city='Orlando, Florida, USA', utc_offset=-5, climate='humid subtropical (hot humid summers, mild winters)', tz='US/Eastern'), 'Rat': dict(country='US', subdiv='DC', city='Washington, D.C., USA', utc_offset=-5, climate='humid subtropical (hot summers, cool winters)', tz='US/Eastern'), 'LCL': dict(country='GB', subdiv='ENG', city='London, United Kingdom', utc_offset=0, climate='temperate oceanic (mild summers, cool winters)', tz='Europe/London')}
COUNTRY_NAME = {'US': 'United States', 'GB': 'United Kingdom'}
UA = 'llm-agents-load-forecasting/0.1 (research; contact via GitHub hirxt0)'

def site_of(building_id):
    name = building_id.split('@')[0]
    return 'LCL' if name.startswith('MAC') else name.split('_')[0]

def anon_id(building_id):
    return f'B{zlib.crc32(building_id.encode()) % 1000:03d}'

@lru_cache(maxsize=1)
def _bdg2_meta():
    m = pd.read_csv(CONTEXT_RAW / 'bdg2_metadata.csv')
    return m.set_index('building_id')

def building_info(building_id, reveal_city=False):
    site = SITES[site_of(building_id)]
    info = {'building_id': anon_id(building_id), 'country': COUNTRY_NAME[site['country']], 'climate': site['climate'], 'timezone': site['tz'], 'units': 'kWh per hour'}
    if site_of(building_id) == 'LCL':
        info['use'] = 'residential household (single smart meter)'
    else:
        row = _bdg2_meta().loc[building_id.split('@')[0]]
        info['use'] = f'non-residential building: {row.primaryspaceusage}' + (f' ({row.sub_primaryspaceusage})' if isinstance(row.sub_primaryspaceusage, str) else '')
        if pd.notna(row.sqm):
            info['floor_area_m2'] = round(float(row.sqm))
        if pd.notna(row.yearbuilt):
            info['year_built'] = int(row.yearbuilt)
    if reveal_city:
        info['city'] = site['city']
    return info

def holiday_name(building_id, day):
    s = SITES[site_of(building_id)]
    cal = holidays.country_holidays(s['country'], subdiv=s['subdiv'], years=day.year)
    return cal.get(day.date())

@lru_cache(maxsize=8)
def _weather_series(site):
    if site == 'LCL':
        j = json.load(open(CONTEXT_RAW / 'weather_london_openmeteo.json'))
        s = pd.Series(j['hourly']['temperature_2m'], index=pd.to_datetime(j['hourly']['time']))
    else:
        w = pd.read_csv(RAW / 'BDG-2' / f'weather_{site}.csv', index_col=0, parse_dates=True)
        s = w['temperature']
    s = s[~s.index.duplicated()].sort_index()
    return s.resample('h').mean().interpolate(limit=6)

def weather(building_id, ts_hist, ts_future, mode, key):
    s = _weather_series(site_of(building_id))
    hist = s.reindex(pd.DatetimeIndex(ts_hist))
    fut = s.reindex(pd.DatetimeIndex(ts_future)).to_numpy(float)
    if mode == 'noisy':
        rng = np.random.default_rng(zlib.crc32(str(key).encode()))
        e = np.empty(len(fut))
        e[0] = rng.normal()
        for h in range(1, len(fut)):
            e[h] = 0.9 * e[h - 1] + np.sqrt(1 - 0.81) * rng.normal()
        idx = pd.DatetimeIndex(ts_future)
        sd = np.linspace(1.0, 2.0, len(fut))
        sd = sd * np.where((idx.hour >= 21) | (idx.hour <= 6), 1.2, 1.0) * np.where(idx.month.isin([12, 1, 2]), 1.2, 1.0)
        fut = fut + e * sd
    return (hist, None if mode == 'none' else fut)

def _cache_get(key):
    p = CACHE / 'news' / f'{zlib.crc32(key.encode()):08x}.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else None

def _cache_put(key, value):
    p = CACHE / 'news' / f'{zlib.crc32(key.encode()):08x}.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(value, open(p, 'w', encoding='utf-8'), ensure_ascii=False)

def _wiki_as_of(title, as_of_utc):
    key = f'{title}|{as_of_utc}'
    cached = _cache_get(key)
    if cached is not None:
        return cached
    q = {'action': 'query', 'prop': 'revisions', 'titles': title, 'rvlimit': 1, 'rvdir': 'older', 'rvstart': as_of_utc, 'rvprop': 'ids|timestamp|content', 'rvslots': 'main', 'format': 'json', 'formatversion': 2}
    url = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(q)
    for attempt in range(3):
        try:
            j = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=30))
            break
        except Exception:
            time.sleep(2 * (attempt + 1))
    else:
        return {'error': 'news service unavailable'}
    page = j['query']['pages'][0]
    if 'revisions' not in page:
        out = {'revision_time': None, 'revid': None, 'text': ''}
    else:
        rev = page['revisions'][0]
        out = {'revision_time': rev['timestamp'], 'revid': rev.get('revid'), 'text': rev['slots']['main']['content']}
        assert rev['timestamp'] <= as_of_utc, 'ревизия новее момента прогноза'
    _cache_put(key, out)
    return out

def _wikitext_to_lines(text):
    lines = []
    for line in text.splitlines():
        if not line.startswith('*'):
            continue
        t = re.sub('\\[\\[(?:[^|\\]]*\\|)?([^\\]]*)\\]\\]', '\\1', line)
        t = re.sub('\\[https?://\\S+\\s*([^\\]]*)\\]', '', t)
        t = re.sub("\\{\\{[^}]*\\}\\}|'''|''|<[^>]+>", '', t).lstrip('* ').strip()
        if len(t) > 20:
            lines.append(t)
    return lines

def news(day, origin, query=None, max_items=25, utc_offset=0):
    day, origin = (pd.Timestamp(day).normalize(), pd.Timestamp(origin))
    if day > origin.normalize():
        return {'error': 'future dates are not available'}
    as_of_ts = origin - pd.Timedelta(hours=utc_offset) - pd.Timedelta(hours=1)
    as_of = as_of_ts.strftime('%Y-%m-%dT%H:%M:%SZ')
    title = f"Portal:Current events/{day.year} {day.strftime('%B')} {day.day}"
    page = _wiki_as_of(title, as_of)
    if 'error' in page:
        return page
    lines = _wikitext_to_lines(page['text'])
    if query:
        words = [w.lower() for w in re.findall('\\w{3,}', query)]
        lines = [l for l in lines if any((w in l.lower() for w in words))]
    return {'date': str(day.date()), 'snapshot_as_of_utc': as_of, 'revision_id': page.get('revid'), 'revision_time_utc': page.get('revision_time'), 'items': lines[:max_items], 'total_matching': len(lines)}
