"""Build a traceable research catalog from indexed original IQON JSON data.

Reads only bounded, CRC-checked archive members; no bulk extraction or archived
program execution. The catalog is separate from production recommendation data.
"""
from pathlib import Path
import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from backend.official_drive_archive import read_indexed_member
from backend.utility_calibration import read, write, digest


class LocalArchive:
    def __init__(self, path):
        self.file = path.open('rb')
        self.size = path.stat().st_size

    def seek(self, offset):
        return self.file.seek(offset)

    def read(self, length):
        return self.file.read(length)

    def close(self):
        self.file.close()


def entries(path):
    with path.open(encoding='utf8') as source:
        for line in source:
            yield json.loads(line)


def ingest(source):
    archive_path = source / 'IQON3000.zip'
    if not archive_path.exists():
        raise ValueError('Complete and verify the original archive first')
    manifest = read(source / 'metadata-acquisition-manifest.json')
    if archive_path.stat().st_size != manifest['archiveBytes']:
        raise ValueError('Original archive size changed')
    target = source / 'catalog.sqlite'
    if (source / 'catalog-report.json').exists():
        print('Completed original-source catalog preserved', flush=True)
        return
    ids = set(read(source / 'features/itemIds.json'))
    db = sqlite3.connect(target)
    db.executescript('''
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS products(item_id TEXT PRIMARY KEY, name TEXT, category_color TEXT,
            colors TEXT, brands TEXT, materials TEXT, category_path TEXT, source_outfit TEXT);
        CREATE TABLE IF NOT EXISTS images(item_id TEXT PRIMARY KEY, archive_entry TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS outfit_evidence(set_id TEXT PRIMARY KEY, user_id TEXT, item_ids TEXT, source_entry TEXT);
        CREATE TABLE IF NOT EXISTS pair_evidence(top_id TEXT, bottom_id TEXT, set_id TEXT,
            PRIMARY KEY(top_id,bottom_id,set_id));
        CREATE INDEX IF NOT EXISTS pairs_by_top ON pair_evidence(top_id);
        CREATE INDEX IF NOT EXISTS pairs_by_bottom ON pair_evidence(bottom_id);
        CREATE VIRTUAL TABLE IF NOT EXISTS product_text USING fts5(item_id UNINDEXED,name,category_color,colors,brands,materials);
        CREATE TABLE IF NOT EXISTS rejected_metadata(path TEXT PRIMARY KEY,error TEXT);
        CREATE TABLE IF NOT EXISTS ingest_state(key TEXT PRIMARY KEY,value TEXT);
    ''')
    for entry in entries(source / 'source-image-index.jsonl'):
        db.execute('INSERT OR IGNORE INTO images VALUES (?,?)', (entry['itemId'], json.dumps(entry)))
    known = defaultdict(set)
    for split in ['train', 'valid', 'test']:
        with (source / f'data/{split}.csv').open() as incoming:
            for line in incoming:
                _, top, bottom, _ = line.strip().split(',')
                known[top].add(bottom)
    archive = LocalArchive(archive_path)
    seen = {r[0] for r in db.execute('SELECT item_id FROM products')}
    state = db.execute("SELECT value FROM ingest_state WHERE key='lastIndex'").fetchone()
    last_index = int(state[0]) if state else -1
    counts, categories = Counter(), Counter()
    try:
        for number, entry in enumerate(entries(source / 'source-metadata-index.jsonl')):
            if number <= last_index:
                continue
            try:
                data = json.loads(read_indexed_member(archive, entry))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                db.execute('INSERT OR REPLACE INTO rejected_metadata VALUES (?,?)', (entry['path'], type(error).__name__))
                continue
            items = data.get('items', [])
            if not isinstance(items, list):
                raise ValueError('Unexpected original item schema')
            item_ids = {str(item.get('itemId')) for item in items} & ids
            if not item_ids:
                continue
            set_id = str(data['setId'])
            user_id = entry['path'].split('/')[1]
            db.execute('INSERT OR IGNORE INTO outfit_evidence VALUES (?,?,?,?)',
                       (set_id, user_id, json.dumps(sorted(item_ids)), entry['path']))
            for item in items:
                pid = str(item.get('itemId'))
                if pid not in ids or pid in seen:
                    continue
                seen.add(pid)
                def field(key):
                    value = item.get(key, [])
                    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
                name = str(item.get('itemName', ''))
                category = str(item.get('category x color', ''))
                colors, brands, materials = field('colors'), field('brands'), field('options')
                db.execute('INSERT INTO products VALUES (?,?,?,?,?,?,?,?)',
                           (pid, name, category, colors, brands, materials, field('breadcrumb'), set_id))
                db.execute('INSERT INTO product_text VALUES (?,?,?,?,?,?)', (pid, name, category, colors, brands, materials))
                counts['products'] += 1
                counts['withName'] += bool(name)
                counts['withCategoryColor'] += bool(category)
                categories[category.split(' × ')[0]] += 1
            for top in item_ids:
                for bottom in known.get(top, set()) & item_ids:
                    db.execute('INSERT OR IGNORE INTO pair_evidence VALUES (?,?,?)', (top, bottom, set_id))
            counts['outfits'] += 1
            if number % 10000 == 0:
                db.execute("INSERT OR REPLACE INTO ingest_state VALUES ('lastIndex',?)", (str(number),))
                db.commit()
                print(json.dumps({'metadataRead': number, **dict(counts)}), flush=True)
        db.commit()
        pairs = db.execute('SELECT COUNT(*) FROM (SELECT DISTINCT top_id,bottom_id FROM pair_evidence)').fetchone()[0]
        evidence = db.execute('SELECT COUNT(*) FROM pair_evidence').fetchone()[0]
        counts = {'products': db.execute('SELECT COUNT(*) FROM products').fetchone()[0],
                  'outfits': db.execute('SELECT COUNT(*) FROM outfit_evidence').fetchone()[0],
                  'withName': db.execute("SELECT COUNT(*) FROM products WHERE name<>''").fetchone()[0],
                  'withCategoryColor': db.execute("SELECT COUNT(*) FROM products WHERE category_color<>''").fetchone()[0],
                  'malformedMetadataRecords': db.execute('SELECT COUNT(*) FROM rejected_metadata').fetchone()[0]}
        categories = Counter()
        for category, in db.execute('SELECT category_color FROM products'):
            categories[category.split(' × ')[0]] += 1
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        report = {'status': 'research_catalog_acquired', **dict(counts),
            'featureBackedProducts': len(ids), 'missingProducts': sorted(ids - seen),
            'distinctPublicPairsWithOutfitProvenance': pairs, 'outfitPairEvidenceRecords': evidence,
            'categories': dict(categories.most_common()), 'sourceArchiveSha256': manifest['archiveSha256'],
            'productionIndexUpdated': False, 'unobservedMeaning': 'no recorded pair, not incompatible'}
        write(source / 'catalog-report.json', report)
        print(json.dumps(report, ensure_ascii=True), flush=True)
    finally:
        archive.close()
        db.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    ingest(p.parse_args().source)
