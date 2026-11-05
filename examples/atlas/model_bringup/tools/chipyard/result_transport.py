"""Explicit UART transport for new full-host result records."""
import json


def integer(value):
    return int(value, 0)


def add_arguments(parser):
    parser.add_argument('--uart-kind', choices=('sifive', 'ns16550'), default='sifive')
    parser.add_argument('--uart-base', type=integer)
    parser.add_argument('--uart-divisor', type=integer)


def write_config(out, args):
    kind, base, divisor = args.uart_kind, args.uart_base, args.uart_divisor
    if base is None:
        if kind != 'sifive':
            raise ValueError('NS16550 UART requires an explicit base')
        base = 0x10020000
    if not isinstance(base, int) or isinstance(base, bool) or not 0 < base < 2**64 or base % 4:
        raise ValueError('UART base must be a positive aligned physical address')
    if divisor is not None and (not isinstance(divisor, int) or isinstance(divisor, bool)
                                or not 0 < divisor < 2**16):
        raise ValueError('UART divisor must fit a nonzero 16-bit register')
    profile = dict(kind=kind, base=base, divisor=divisor, output='physical UART MMIO',
                   baud='hardware configuration preserved' if divisor is None else 'explicit divisor',
                   htif=False, hardware_execution=False)
    lines = ['#ifndef ATLAS_RESULT_TRANSPORT_H', '#define ATLAS_RESULT_TRANSPORT_H',
             '#include "atlas_result.h"',
             'typedef struct { uintptr_t base; } atlas_result_uart;',
             'static inline void atlas_result_uart_init(atlas_result_uart *uart) {',
             f'  uart->base = UINT64_C({base});']
    if kind == 'sifive':
        if divisor is not None:
            lines.append(f'  *(volatile uint32_t *)(uart->base + 24) = {divisor};')
        lines.append('  *(volatile uint32_t *)(uart->base + 8) = 1;')
    elif divisor is not None:
        lines += ['  volatile uint8_t *regs = (volatile uint8_t *)uart->base;',
                  '  uint8_t lcr = regs[3]; regs[3] = lcr | 0x80;',
                  f'  regs[0] = {divisor & 255}; regs[1] = {divisor >> 8};',
                  '  regs[3] = lcr & 0x7f;']
    lines += ['}', 'static inline void atlas_result_uart_putchar(void *context, char value) {',
              '  atlas_result_uart *uart = (atlas_result_uart *)context;']
    if kind == 'sifive':
        lines += ['  volatile uint32_t *tx = (volatile uint32_t *)uart->base;',
                  '  while (*tx & UINT32_C(0x80000000)) { }', '  *tx = (uint8_t)value;']
    else:
        lines += ['  volatile uint8_t *regs = (volatile uint8_t *)uart->base;',
                  '  while (!(regs[5] & 0x20)) { }', '  regs[0] = (uint8_t)value;']
    lines += ['}', '#endif', '']
    (out / 'atlas_result_transport.h').write_text('\n'.join(lines))
    (out / 'result-transport.json').write_text(json.dumps(profile, indent=2) + '\n')
    return profile
