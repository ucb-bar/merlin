/* Synthetic fixture. */
#include "iree/hal/api.h"
#include <stdint.h>

extern iree_status_t iree_hal_radiance_driver_module_register(
	iree_hal_driver_registry_t *registry);

iree_status_t iree_hal_radiance_driver_create(void) {
	iree_hal_executable_t *exe = NULL;
	iree_hal_command_buffer_t *cb = NULL;
	(void)exe;
	(void)cb;
	return iree_ok_status();
}
