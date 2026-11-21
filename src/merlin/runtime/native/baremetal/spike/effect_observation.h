/* Optional bounded source-stdio observations through the console ABI. */
#ifndef MERLIN_EFFECT_OBSERVATION_H
#define MERLIN_EFFECT_OBSERVATION_H
void merlin_effect_begin(void);
void merlin_effect_return(void);
void merlin_effect_putc(char);
void merlin_effect_puts(const char *);
void merlin_effect_failure_inputs(void);
void merlin_effect_abort(void) __attribute__((noreturn));
void merlin_effect_format_error(void) __attribute__((noreturn));
#endif
