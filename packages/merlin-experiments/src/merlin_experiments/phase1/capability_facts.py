"""Scheduling and memory capability states established by selected RTL facts and the ordering record.

The optimization catalogue offers a mechanism only when every capability it requires is
``supported``; an ``unknown`` one turns it into a discovery question. Several of those
capabilities are properties the RTL facts (``timing_structure``) and the derived command-ordering
record (:mod:`merlin.targetgen.rtl.command_ordering`) already describe: which command queues the
dispatch unit holds, whether a younger command proceeds before an older one completes, which
address dependencies the hardware tracks, which configuration a later command reads, and which
local rows one command covers.

Each derivation below names the one fact that settles it, and returns that fact as provenance.
No fact leaves the capability ``unknown`` with the reason; ``unsupported`` is returned only when a
fact excludes the capability. Nothing is read from a target name, a timing measurement or a
candidate compiler; the ordering record is used only when its pin binds the exact facts and
contract this selection reads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

#: The capabilities this module decides, in report order.
DERIVED = (
    'scheduling.queues',
    'scheduling.finite_queues',
    'scheduling.async_issue',
    'scheduling.multiple_outstanding',
    'scheduling.dependency_tokens',
    'scheduling.persistent_configuration',
    'memory.resident_operands',
    'memory.strided_local_rows',
)
_ORDERED = ('derived', 'partial')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _found(state, source, fact, basis):
    return {'status': state, 'source': source, 'fact': fact, 'basis': basis}


def _missing(reason):
    return {'status': 'unknown', 'reason': reason}


# ------------------------------------------------------------------------------------- inputs
def ordering_input(pin=None, *, facts_document=None, facts_path=None, contract_path=None):
    """(ordering record or None, provenance). An explicit selection pin is used only when its bytes,
    its record digest and its base facts/contract match the inputs this selection reads; otherwise
    the record is withheld (never a fallback) and the reason recorded. Without a pin, a derived
    ``command_ordering`` section of the facts themselves is used."""
    if pin:
        from merlin.targetgen.rtl.command_ordering_selection import SCHEMA

        if isinstance(pin, (str, Path)):
            pin = {'path': str(pin), 'sha256': _sha(pin)}
        path = Path(pin.get('path') or '')
        if not path.is_file() or _sha(path) != pin.get('sha256'):
            return None, {'selection': dict(pin), 'status': 'unavailable',
                          'reason': 'the ordering selection is missing or its bytes changed since it was pinned'}
        doc = json.loads(path.read_bytes())
        record = doc.get('ordering')
        provenance = {'selection': {'path': str(path), 'sha256': pin['sha256']},
                      'ordering_sha256': doc.get('ordering_sha256')}
        checks = [
            (doc.get('schema') == SCHEMA, f'schema is not {SCHEMA}'),
            (isinstance(record, dict) and _digest(record) == doc.get('ordering_sha256'),
             'the ordering record does not match its ordering_sha256'),
            (facts_path is not None and Path(facts_path).is_file()
             and (doc.get('base_facts') or {}).get('sha256') == _sha(facts_path),
             'the selection was derived from other RTL facts than the ones staged here'),
            (contract_path is None or (doc.get('base_contract') or {}).get('sha256') == _sha(contract_path),
             'the selection was derived against another target contract'),
            (isinstance(record, dict) and record.get('status') in _ORDERED, 'the ordering record is not derived'),
        ]
        for passed, reason in checks:
            if not passed:
                return None, dict(provenance, status='unavailable', reason=reason)
        return record, dict(provenance, status=record['status'])
    record = ((facts_document or {}).get('facts') or {}).get('command_ordering')
    if isinstance(record, dict) and record.get('status') in _ORDERED:
        return record, {'source': 'facts.command_ordering', 'status': record['status'],
                        'ordering_sha256': _digest(record)}
    return None, {'status': 'unavailable', 'reason': 'no ordering selection and no derived facts.command_ordering'}


# ------------------------------------------------------------------------------------- derivations
def _classes(facts):
    block = ((facts.get('timing_structure') or {}).get('command_classes')) or {}
    if block.get('status') != 'derived':
        return {}
    return {name: row for name, row in (block.get('classes') or {}).items() if isinstance(row, dict)}


def _entries(row):
    entries = row.get('entries') or {}
    value = entries.get('value')
    return value if entries.get('status') == 'derived' and type(value) is int and value > 0 else None


def queues(facts, ordering):
    for name, row in sorted(_classes(facts).items()):
        if (entries := _entries(row)) is not None:
            module = ((facts.get('timing_structure') or {}).get('command_classes') or {}).get('dispatch_module')
            return _found('supported', 'rtl_facts', f'timing_structure.command_classes.classes.{name}.entries',
                          f'dispatch module {module} holds {entries} entries for command class {name}')
    named = sorted({row.get('queue') for row in (ordering or {}).get('funct_queue') or () if row.get('queue')})
    if named:
        return _found('supported', 'command_ordering', 'command_ordering.funct_queue',
                      f'the dispatch unit presents decoded commands at the issue ports of queues {named}')
    return _missing('no derived dispatch class entries and no ordering record naming a queue')


def finite_queues(facts, ordering):
    for name, row in sorted(_classes(facts).items()):
        if (entries := _entries(row)) is not None:
            return _found('supported', 'rtl_facts', f'timing_structure.command_classes.classes.{name}.entries',
                          f'command class {name} has a finite dispatch table of {entries} entries')
    return _missing('no derived, finite dispatch entry count')


def async_issue(facts, ordering):
    if ordering is None:
        return _missing('needs the command-ordering record: whether a younger command issues before an '
                        'older one completes is a dispatch property the timing structure does not state')
    same = ordering.get('same_queue') or {}
    for queue, row in sorted(same.items()):
        if row.get('release') == 'issue':
            return _found('supported', 'command_ordering', f'command_ordering.same_queue.{queue}.release',
                          f'queue {queue} issues a younger command once the older one has issued, '
                          'before it completes')
    cross = ordering.get('cross_queue') or ()
    for row in cross:
        for rows in ('disjoint', 'overlap'):
            if row.get(rows) in ('issue', 'none'):
                return _found('supported', 'command_ordering',
                              f"command_ordering.cross_queue[{row['older']} -> {row['younger']}, {row['space']}]",
                              f"a younger {row['younger']} on {rows} rows issues before the older "
                              f"{row['older']} completes ({row[rows]})")
    releases = [row.get('release') for row in same.values()] + [
        value for row in cross for value in (row.get('overlap'), row.get('disjoint'))]
    if (ordering.get('status') == 'derived' and not ordering.get('unknown') and releases
            and all(value == 'completion' for value in releases)):
        return _found('unsupported', 'command_ordering', 'command_ordering.same_queue, cross_queue',
                      'every younger command waits for the older one to complete')
    return _missing('the ordering record shows no command released before an older one completes')


def multiple_outstanding(facts, ordering):
    clients = (((facts.get('timing_structure') or {}).get('memory_clients')) or {}).get('clients') or {}
    slots = {name: ((row.get('in_flight') or {}).get('value'), (row.get('in_flight') or {}).get('status'))
             for name, row in clients.items() if isinstance(row, dict)}
    if slots and all(status == 'derived' and value == 1 for value, status in slots.values()):
        return _found('unsupported', 'rtl_facts', 'timing_structure.memory_clients.clients.*.in_flight',
                      'every memory client tracks a single outstanding transaction')
    if ordering is None:
        return _missing('needs the command-ordering record: whether a second transfer command starts '
                        'before the first completes is a dispatch property')
    for queue, row in sorted((ordering.get('completion_order') or {}).items()):
        if row.get('order') == 'out_of_order':
            return _found('supported', 'command_ordering', f'command_ordering.completion_order.{queue}',
                          f"queue {queue} starts a younger transfer before the older one completes and its "
                          f"memory client tracks several transactions ({'; '.join(row.get('clients') or ())})")
    return _missing('no queue both releases at issue and reaches a memory client with several transaction slots')


def dependency_tokens(facts, ordering):
    if ordering is None:
        return _missing('needs the command-ordering record: which dependencies the dispatch tracks')
    tracked = [row for row in ordering.get('cross_queue') or () if row.get('tracked') is True]
    if tracked:
        row = tracked[0]
        pairs = sorted({(r['older'], r['younger']) for r in tracked})
        return _found('supported', 'command_ordering',
                      f"command_ordering.cross_queue[{row['older']} -> {row['younger']}, {row['space']}].tracked",
                      f'the dispatch holds a younger command until an overlapping older command of another queue '
                      f'completes and releases it when disjoint ({len(pairs)} tracked command pairs)')
    return _missing('the ordering record shows no hardware-tracked dependency between queues')


def _span_terms(extent):
    """(name, term) for each span term of an extent that is read from a configuration code."""
    blocks = extent.get('blocks') or {}
    for name, term in (('scale', extent.get('scale')), ('blocks.stride', blocks.get('stride'))):
        if isinstance(term, dict) and type(term.get('code')) is int and 'reset' in term:
            yield name, term


def persistent_configuration(facts, ordering):
    if ordering is None:
        return _missing('needs the command-ordering record: whether a later command reads latched configuration')
    config = ((ordering.get('classes') or {}).get('config')) or {}
    by_code = config.get('by_code') or {}
    if type(config.get('funct')) is not int or not by_code:
        return _missing('the ordering record decodes no separate configuration command')
    for key, extent in sorted((ordering.get('extents') or {}).items()):
        if not isinstance(extent, dict) or extent.get('status') != 'derived' or not extent.get('verified'):
            continue
        for name, term in _span_terms(extent):
            setter = by_code.get(str(term['code']))
            if setter:
                return _found('supported', 'command_ordering', f'command_ordering.extents.{key}.{name}',
                              f"the rows {key} covers depend on a field configuration command {setter} latches "
                              f"({term.get('operand')}[{term.get('offset')}:+{term.get('width')}], reset "
                              f"{term.get('reset')}), verified at fresh points: configuration persists "
                              'across commands')
    return _missing('no verified extent reads a value latched by a configuration command')


def _accesses(ordering):
    return {key: row for key, row in (ordering.get('accesses') or {}).items() if isinstance(row, dict)}


def resident_operands(facts, ordering):
    spaces = {row.get('name') for row in facts.get('memory_address_spaces') or ()
              if isinstance(row, dict) and row.get('view_kind') == 'address_space'
              and type(row.get('bytes')) is int and row['bytes'] > 0}
    if not spaces:
        return _missing('no positive-byte local address space in the facts')
    if ordering is None:
        return _missing('needs the command-ordering record: whether a separate command reads rows another wrote')
    accesses = _accesses(ordering)
    for row in ordering.get('cross_queue') or ():
        older, younger = accesses.get(row.get('older')), accesses.get(row.get('younger'))
        if (older and younger and older.get('direction') == 'write' and younger.get('direction') == 'read'
                and older.get('class') != younger.get('class') and row.get('space') in spaces
                and row.get('overlap') == 'completion'):
            return _found('supported', 'command_ordering',
                          f"command_ordering.cross_queue[{row['older']} -> {row['younger']}, {row['space']}]",
                          f"a separate {younger['class']} command reads {row['space']} rows an earlier "
                          f"{older['class']} wrote, after its completion: local operands stay resident "
                          'between commands')
    return _missing('no ordered write-then-read of the same local rows by two command classes')


def strided_local_rows(facts, ordering):
    if ordering is None:
        return _missing('needs the command-ordering record: which local rows one transfer command covers')
    accesses = _accesses(ordering)
    reads = {row.get('class') for row in accesses.values() if row.get('direction') == 'read'}
    for key, extent in sorted((ordering.get('extents') or {}).items()):
        access = accesses.get(key) or {}
        blocks = (extent or {}).get('blocks') or {}
        stride = blocks.get('stride') or {}
        if (access.get('direction') != 'write' or access.get('class') in reads
                or extent.get('status') != 'derived' or not extent.get('verified')
                or type(blocks.get('block')) is not int or blocks['block'] <= 0
                or not isinstance(blocks.get('cols'), dict) or not isinstance(extent.get('rows'), dict)
                or not (type(stride.get('constant')) is int or type(stride.get('code')) is int)):
            continue
        law = (f"constant {stride['constant']}" if 'constant' in stride
               else f"configuration code {stride['code']}")
        return _found('supported', 'command_ordering', f'command_ordering.extents.{key}.blocks',
                      f"one {access['class']} transfer writes rows x column blocks of {blocks['block']} columns "
                      f"at a local block stride ({law}), verified at fresh points")
    return _missing('no verified transfer-in extent with column blocks at a local block stride')


DERIVATIONS = {
    'scheduling.queues': queues,
    'scheduling.finite_queues': finite_queues,
    'scheduling.async_issue': async_issue,
    'scheduling.multiple_outstanding': multiple_outstanding,
    'scheduling.dependency_tokens': dependency_tokens,
    'scheduling.persistent_configuration': persistent_configuration,
    'memory.resident_operands': resident_operands,
    'memory.strided_local_rows': strided_local_rows,
}


def derive(document, ordering=None, *, ordering_provenance=None):
    """(values, evidence, provenance): ``values`` holds only fact-settled states; ``provenance`` holds
    every decided capability's fact and every undecided one's reason."""
    facts = (document or {}).get('facts') or {}
    values, evidence, provenance = {}, {}, {}
    for trait, rule in DERIVATIONS.items():
        found = rule(facts, ordering)
        if found['status'] != 'unknown' and found['source'] == 'command_ordering' and ordering_provenance:
            found['input'] = ordering_provenance
        provenance[trait] = found
        if found['status'] != 'unknown':
            values[trait] = found['status']
            evidence[trait] = f"{found['fact']}: {found['basis']}"
    return values, evidence, provenance
