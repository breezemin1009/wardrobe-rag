"""Keep a retrieved paper passage with its continuation on the same PDF page."""

def with_research_context(records, chunks, limit):
    positions = {chunk['id']: index for index, chunk in enumerate(chunks)}
    result, seen = [], set()

    def append(record):
        if record['id'] not in seen and len(result) < limit:
            result.append(record)
            seen.add(record['id'])

    for record in records:
        index = positions.get(record['id'])
        full = chunks[index] if index is not None else record
        append({**record, 'text':full['text']})
        if record.get('kind') != 'paper' or index is None or index+1 >= len(chunks):
            continue
        neighbor = chunks[index+1]
        if (neighbor.get('kind') == 'paper' and neighbor.get('sourceId') == record.get('sourceId')
                and neighbor.get('locator') == record.get('locator') and neighbor.get('currentScope')):
            append({**neighbor, 'retrieval':record.get('retrieval',{}), 'contextOf':record['id']})
    return result
